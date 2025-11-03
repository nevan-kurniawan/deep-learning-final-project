"""
This module defines the ModifiedModel class.

The module defines the ModifiedModel class as the primary orchestrator, and
the MoE_LoRA class is the implementation for the MoE-LoRA architecture.
The MoE_LoRA class uses the Router and BatchedLoRA
module as part of its logic.
"""

import torch
from torch import nn
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model
import math

class Router(nn.Module):
    """
    Implements a top-k gating mechanism (router) for a Mixture-of-Experts layer.

    This module takes token embeddings as input and outputs the indices and weights
    for the top-k experts, along with an auxiliary loss for load balancing.
    """
    def __init__(self, input_dim: int, num_experts: int, k: int = 2):
        """
        Initializes the Router module.
        Args:
            input_dim: The embedding dimension of the input tokens.
            num_experts: Total number of experts (N).
            k: Total number of experts to route tokens to (default is 2).
        """
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        self.gating_network = nn.Linear(input_dim, self.num_experts)

    def compute_auxiliary_loss(self, router_probs: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the auxiliary load-balancing loss.

        The auxiliary load-balancing loss is used to prevent expert collapse, where the model favors a few experts,
        and learns to route only to those experts, leaving the rest under-trained. This loss ensures the tokens are routed
        relatively evenly to all experts.
        Args:
            router_probs: The softmax probabilities output by the gating network. Shape: [T, N] (num_tokens, num_experts).
            expert_mask: A boolean mask indicating which tokens were dispatched to which expert. Shape: [T, N] (num_tokens, num_experts).

        Returns:
            auxiliary_loss: Scalar auxiliary loss.
        """
        if router_probs is None or expert_mask is None:
            return torch.tensor(0.0)  # No loss if no routing happened

        # Calculate the fraction of tokens dispatched to each expert.
        # Shape: [N]
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        # Calculate the average router probability for each expert.
        # Shape: [N]
        router_prob_per_expert = torch.mean(router_probs, dim=0)

        # The loss is the dot product of these two quantities, scaled by the number of experts.
        auxiliary_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)
        return auxiliary_loss

    def forward(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes expert assignments and gating weights for a batch of tokens.
        Args:
            x_flat: The flattened input token embeddings. Shape: [T, D_in] (num_tokens, input_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                top_k_weights: The weights for the top-k experts. Shape: [T, k]
                top_k_indices: The indices of the top-k experts. Shape: [T, k]
                auxiliary_loss: The scalar load-balancing loss.
        """

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

class BatchedLoRA(nn.Module):
    """
    Implements a batched version of Low-Rank Adaptation (LoRA) for Mixture-of-Experts.

    This module holds N sets of LoRA A and B matrices and computes the LoRA update
    for all N experts in parallel for a given input batch.
    """

    def __init__(self, input_dim: int, output_dim: int, rank: int, alpha: int, num_experts: int):
        """
        Initializes the BatchedLoRA layer.
        Args:
            input_dim: Input dimension (D_in) of the layer being adapted.
            output_dim: Output dimension (D_out) of the layer being adapted.
            rank: The rank 'r' of the LoRA decomposition.
            alpha: The LoRA scaling factor.
            num_experts: The total number of expert LoRA matrices (N).
        """
        super().__init__()
        # Define trainable LoRA parameters for all experts
        self.A_weights = nn.Parameter(torch.empty(num_experts, rank, input_dim))
        self.B_weights = nn.Parameter(torch.empty(num_experts, output_dim, rank))

        # LoRA scaling factor
        self.scaling = alpha/rank

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the LoRA matrices"""
        # A weights are initialized with kaiming uniform.
        # a is chosen based on the same initialization bound of a standard nn.Linear layer
        nn.init.kaiming_uniform_(self.A_weights, a=math.sqrt(5))
        # B weights are initialized with zero matrices
        nn.init.zeros_(self.B_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the LoRA update delta for all experts in parallel.

        Args:
            x: Input tensor flattened to token level. Shape: [T, D_in] (num_tokens, input_dim)

        Returns:
            torch.Tensor: The LoRA update deltas for all N experts and T tokens.
                Shape: [T, N, D_out] (num_tokens, num_experts, Output_Dim)
        """
        # Down-projection: Applies all N expert A matrices to T tokens
        # 'td, nrd -> tnr' maps [T, D_in] and [N, r, D_in] to [T, N, r]
        a_downproject = torch.einsum('td, nrd -> tnr', x, self.A_weights)

        # Up-projection: Applies all N expert B matrices
        # 'tnr, nor -> tno' maps [T, N, r] and [N, D_out, r] to [T, N, D_out]
        b_upproject = torch.einsum('tnr, nor-> tno', a_downproject, self.B_weights)

        return b_upproject * self.scaling

class MoE_LoRA(nn.Module):
    """
    Implements a single attention block layer modified with a Mixture-of-Experts (MoE) architecture. The experts are low-rank (LoRA) adapters.

    This implementation uses a vectorized compute-all then select design. All experts are applied to all tokens, computed in parallel,
    then combined via a weighted sum from the router that acts as a gating mechanism.
    """
    def __init__(self, base_layer: nn.Module, num_experts: int, lora_rank: int, lora_alpha: int, k: int = 2):
        """
        Initializes the MoE_LoRA layer.

        Args:
            base_layer: The original PyTorch module (e.g., a ViT Attention block)
                                    that this layer replaces and whose parameters will be frozen.
            num_experts: The total number of expert networks (N) in the pool. (Each one is a set of A and B LoRA parameters)
            lora_rank: The rank 'r' for the low-rank decomposition in LoRA experts.
            lora_alpha: The LoRA scaling factor (alpha).
            k: The number of experts to route each token to. Defaults to 2.
        """
        super().__init__()
        ### Original attributes of the modified block
        self.base_layer = base_layer
        self.num_heads = base_layer.num_heads
        self.scale = base_layer.scale
        self.attn_drop = base_layer.attn_drop
        self.proj_drop = base_layer.proj_drop

        ### Original qkv and projection of the modified block
        self.frozen_qkv = base_layer.qkv
        self.frozen_proj = base_layer.proj

        ### MoE components
        self.num_experts = num_experts
        input_dim = self.base_layer.qkv.in_features
        output_dim = self.base_layer.qkv.out_features
        self.router = Router(input_dim, num_experts, k)
        self.experts = BatchedLoRA(
            num_experts=num_experts,
            input_dim=input_dim,
            output_dim=output_dim,
            rank=lora_rank,
            alpha=lora_alpha,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Defines the shape and data flow for the MoE-LoRA layer.

        Args:
            x (torch.Tensor): Input tensor.
                Shape: [B, S, D_in] (Batch, Sequence, Input Dim)
            **kwargs: Accepts additional keyword arguments for compatibility with
                parent modules, though they are not used in this layer.
        Returns:
            torch.Tensor: Output tensor of the attention block.
                Shape: [B, S, D_in] (Batch, Sequence, Input Dim)
        """
        ### 1. Setup and dimension definitions
        batch_size, seq_len, input_dim = x.shape
        num_tokens = batch_size * seq_len
        x_flat = x.view(num_tokens, input_dim)

        ### 2. Parallel forward passes
        # 2a. Main path (Frozen base QKV projection)
        qkv_base = self.frozen_qkv(x) # shape = [B, S, D_in]

        # 2b. Parallel MoE-LoRA path
        # Outputs: top-k weights [T, k], top-k indices [T, k], scalar loss
        top_k_weights, top_k_indices, auxiliary_loss = self.router(x_flat)
        # Store the auxiliary loss for forward hook collection
        self.aux_loss = auxiliary_loss

        # Compute all expert outputs (projects D_in -> 3*D_in (D_out))
        expert_output = self.experts(x_flat) # shape = [T, N, 3*D_in]

        ### 3. Gating and combination
        # Create sparse dispatch weights [T, N] from top-k outputs
        routing_matrix = F.one_hot(top_k_indices, self.num_experts) #Shape = [T, k, N]

        # Apply routing weights via broadcasting = [T, k, N] * [T, k, 1] -> [T, k, N]
        scaled_routing_matrix = routing_matrix * top_k_weights.unsqueeze(dim=-1)
        dispatch_tensor = scaled_routing_matrix.sum(dim=1) #Shape = [T, N]

        # Select and combine expert outputs via weighted sum using einsum
        # 'tno, tn -> to' combines [T, N, D_out] and [T, N] into [T, D_out]
        sparse_scaled_expert_output = torch.einsum('tno, tn-> to', expert_output, dispatch_tensor)

        ### 4. Add LoRA delta from the LoRA path to the frozen base QKV projection
        qkv = qkv_base + sparse_scaled_expert_output.view(batch_size, seq_len, -1)

        ### 5. Standard Multi-Head attention logic
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, input_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q@k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, input_dim)
        x = self.frozen_proj(x)
        x = self.proj_drop(x)

        return x

class ModifiedModel(nn.Module):
    """
    The top-level class responsible for orchestrating the components.
    """
    def __init__(self, base_model: nn.Module, num_classes: int, model_config: dict):
        """
        Initializes the ModifiedModel class.
        Args:
            base_model: The pre-trained base model to be modified (either a DeiT or ViT).
            num_classes: Number of classes in the dataset.
            model_config: The model configuration dictionary taken from the config files.
        """
        super().__init__()
        self.base_model = base_model

        mode = model_config.get('mode')
        lora_rank = model_config.get('lora_rank')
        lora_alpha = model_config.get('lora_alpha')
        num_experts = model_config.get('num_experts')
        k = model_config.get('k')
        target_blocks_indices = model_config.get('target_blocks_indices', [])

        # freeze all parameters first
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Head replacement
        in_features = self.base_model.head.in_features
        self.base_model.head = nn.Linear(in_features, num_classes)

        # Conditional setup based on mode
        if mode == 'full_finetune':
            # Unfreeze everything
            for param in self.base_model.parameters():
                param.requires_grad = True

        elif mode == 'linear_probe':
            for param in self.base_model.head.parameters():
                param.requires_grad = True

        elif mode == 'lora':
            target_modules = [f"blocks.{i}.attn.qkv" for i in target_blocks_indices]
            lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
            self.base_model = get_peft_model(self.base_model, lora_config)
            for param in self.base_model.head.parameters():
                param.requires_grad = True

        elif mode == 'moe_lora':
            for block_index in target_blocks_indices:
                original_attention_layer = self.base_model.blocks[block_index].attn
                self.base_model.blocks[block_index].attn = MoE_LoRA(
                    base_layer=original_attention_layer,
                    num_experts=num_experts,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    k=k
                )
            for param in self.base_model.head.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.collected_auxiliary_loss = [] # store auxiliary loss for the forward hook to gather

        def collect_auxiliary_loss(module, _, __):
            """
            The forward hook function to collect the stateful auxiliary loss from an nn.Module.
            Args:
                module: The module which would have its auxiliary loss collected.
                _: Ignored input
                __: Ignored output
            """
            if hasattr(module, 'aux_loss'):
                self.collected_auxiliary_loss.append(module.aux_loss)

        for block in self.base_model.blocks:
            if isinstance(block.attn, MoE_LoRA):
                block.attn.register_forward_hook(collect_auxiliary_loss)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the forward pass for the entire model.
        Args:
            x: The raw input tensor.

        Returns:
            logits: The logits of the model.
            total_aux_loss: The auxiliary loss (if it uses MoE_LoRA). Else, 0.
        """
        self.collected_auxiliary_loss.clear()
        logits = self.base_model(x)

        # Conditional check for if an auxiliary loss exists. If not, pass 0 as the auxiliary loss.
        if self.collected_auxiliary_loss:
            total_aux_loss = torch.stack(self.collected_auxiliary_loss).mean()
        else:
            total_aux_loss = torch.tensor(0.0, device=logits.device)

        return logits, total_aux_loss