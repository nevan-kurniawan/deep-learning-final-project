import torch
import timm
from torch import nn
from torch.nn import functional as F
from peft import LoraConfig, TaskType, get_peft_model
import copy

base_model = timm.create_model('deit_tiny_patch16_224', pretrained=True)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["qkv"],
    lora_dropout=0.1,
    bias="none"
)

target_blocks_indices = [8, 9, 10, 11] #last 4 blocks

class Router(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, k: int = 2):
        """
        top-k router for Mixture-of-Experts. This takes an input, then returns

        Parameters
        ----------
        input_dim : int
            The embedding dimension of the input tokens.
        num_experts : int
            The total number of experts available.
        k : int, optional
            The number of experts to route each token to, by default 2.
        """
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        # The gating network is a simple linear layer.
        self.gating_network = nn.Linear(input_dim, self.num_experts)

    def compute_auxiliary_loss(self, router_probs: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the auxiliary load-balancing loss.

        This loss is critical for ensuring that the router sends tokens to all experts
        relatively evenly, preventing a situation where only a few experts are used.

        Parameters
        ----------
        router_probs : torch.Tensor
            The softmax probabilities output by the gating network. Shape: (num_tokens, num_experts).
        expert_mask : torch.Tensor
            A boolean mask indicating which tokens were dispatched to which expert.
            Shape: (num_tokens, num_experts).

        Returns
        -------
        torch.Tensor
            The scalar auxiliary loss value.
        """
        if router_probs is None or expert_mask is None:
            return torch.tensor(0.0)  # No loss if no routing happened

        # Calculate the fraction of tokens dispatched to each expert.
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Calculate the average router probability for each expert.
        router_prob_per_expert = torch.mean(router_probs, dim=0)

        # The loss is the dot product of these two quantities, scaled by the number of experts.
        auxiliary_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)
        return auxiliary_loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes expert assignments and gating weights for a batch of tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input token embeddings. Shape: (batch_size, seq_len, input_dim).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - top_k_weights: The softmax weights for the top-k experts for each token.
            - top_k_indices: The indices of the top-k experts for each token.
        """
        # Flatten the input from (batch, seq, dim) to (total_tokens, dim)
        num_tokens = x.shape[0] * x.shape[1]
        x_flat = x.view(num_tokens, -1)

        # Compute router logits and probabilities
        logits = self.gating_network(x_flat)
        router_probs = F.softmax(logits, dim=-1)

        # Select the top-k experts for each token
        top_k_weights, top_k_indices = torch.topk(router_probs, k=self.k, dim=-1)

        # Create a mask for the auxiliary loss calculation
        expert_mask = F.one_hot(top_k_indices, self.num_experts).sum(dim=1)

        # Compute the auxiliary loss

        self.curr_auxiliary_loss = self.compute_auxiliary_loss(router_probs, expert_mask)

        return top_k_weights, top_k_indices

class LoRA_Expert(nn.Module):
    def __init__(self, input_dim: int, output_dim, rank, alpha):
        super().__init__()
        self.A_param = nn.Linear(input_dim, rank)
        self.B_param = nn.Linear(rank, output_dim)
        self.scaling = alpha/rank

        nn.init.kaiming_uniform_(self.A_param.weight)
        nn.init.zeros_(self.B_param.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B_param(self.A_param(x)) * self.scaling

class MoE_LoRA(nn.Module):
    def __init__(self, base_layer: nn.Module, num_experts: int, lora_rank: int, lora_alpha: int, k: int = 2):
        super().__init__()
        self.base_layer = base_layer
        self.num_heads = base_layer.num_heads
        self.scale = base_layer.scale
        self.frozen_qkv = base_layer.qkv
        self.frozen_proj = base_layer.proj
        self.attn_drop = base_layer.attn_drop
        self.proj_drop = base_layer.proj_drop

        input_dim = self.base_layer.qkv.in_features
        output_dim = self.base_layer.qkv.out_features

        self.router = Router(input_dim, num_experts, k)
        self.experts  = nn.ModuleList([LoRA_Expert(input_dim, output_dim, lora_rank, lora_alpha) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top_k_weights, top_k_indices = self.router(x)
        batch_size = x.shape[0]
        num_tokens = x.shape[1]
        channels = x.shape[2]

        total_tokens = batch_size * num_tokens

        x_flat = x.view(total_tokens, channels)

        qkv_base = self.frozen_qkv(x)

        lora_delta_flat = torch.zeros_like(qkv_base.view(total_tokens, -1))

        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i) #Make a boolean mask with the shape of top_k_indices. If the i matches, it would be 1 (true), and if not, it would be 0 (false)

            token_indices, k_choice_indices = torch.where(mask) #token_indices is a list containing the row where there are 1 (this helps find the row index of the tokens that are sent to expert i), k_choice_indices is a list containing the columns where there are 1 (this helps find if the chosen token is the first or second choice for being sent to expert i).

            if token_indices.shape[0] == 0:
                continue

            selected_weights = top_k_weights[token_indices, k_choice_indices] #Get the weights/confidence for that token for expert i
            selected_tokens = x_flat[token_indices] #Get the embeddings of the token

            expert_output = expert(selected_tokens) #Choose the i expert from the ModuleList, then process that token with that expert.

            scaled_output = expert_output * selected_weights.unsqueeze(-1) #Unsqueeze for broadcasting, to make the shape match. The selected_weights would be broadcasted.

            lora_delta_flat.index_add_(0, token_indices, scaled_output) #This is used to add the output of expert i to the tokens that are processed by that expert.

        qkv = qkv_base + lora_delta_flat.view(batch_size, num_tokens, -1)

        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        attn = (q@k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)

        x = self.frozen_proj(x)
        x = self.proj_drop(x)

        return x

class ModifiedModel(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes: int, lora_rank: int, lora_alpha: int, num_experts: int, k: int = 2):
        super().__init__()
        self.base_model = base_model

        for param in self.base_model.parameters(): #Freeze parameters
            param.requires_grad = False

        in_features = self.base_model.head.in_features

        self.base_model.head = nn.Linear(in_features, num_classes)

        for block_index in target_blocks_indices:
            original_attention_layer = self.base_model.blocks[block_index].attn
            self.base_model.blocks[block_index].attn = MoE_LoRA(
                base_layer=original_attention_layer,
                num_experts=num_experts,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                k=k
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.base_model(x)

        aux_loss = []

        for block in self.base_model.blocks:
            if isinstance(block.attn, MoE_LoRA): #If it's an instance of a MoE_LoRA object and is also an attention block. That means if it's a modified block.
                aux_loss.append(block.attn.router.curr_auxiliary_loss)

        total_aux_loss = torch.stack(aux_loss).mean()

        return logits, total_aux_loss