import torch
import timm
from torch import nn
from torch.nn import functional as F
from peft import LoraConfig, TaskType, get_peft_model

base_model = timm.create_model('deit_tiny_patch16_224', pretrained=True)

# Takes the last four self-attention qkv layers
target_modules = [
    "blocks.8.attn.qkv",
    "blocks.9.attn.qkv",
    "blocks.10.attn.qkv",
    "blocks.11.attn.qkv"
]

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none"
)


class router(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, k: int = 2):
        """
        A simple top-k router for a Mixture-of-Experts model.

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
        # This is from the Switch Transformer paper.
        auxiliary_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)
        return auxiliary_loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes expert assignments and gating weights for a batch of tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input token embeddings. Shape: (batch_size, seq_len, input_dim).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - top_k_weights: The softmax weights for the top-k experts for each token.
            - top_k_indices: The indices of the top-k experts for each token.
            - auxiliary_loss: The load-balancing auxiliary loss.
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
        auxiliary_loss = self.compute_auxiliary_loss(router_probs, expert_mask)

        return top_k_weights, top_k_indices, auxiliary_loss