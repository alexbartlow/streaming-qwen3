"""
Streaming MoE Forward Pass with Expert Caching.

Patches the model to use cached experts from VRAM/RAM with graceful
substitution on cache miss (exploiting fungibility of middle layers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loader import ModelLoader, ExpertWeights
from .cache import ExpertCacheManager


class StreamingMoELayer(nn.Module):
    """
    Drop-in replacement for Qwen3's MoE block using cached experts.

    Fetches from cache, which may substitute a different expert on miss
    for fungible layers (near-zero quality impact based on KL measurements).
    """

    def __init__(
        self,
        layer_idx: int,
        original_moe: nn.Module,
        cache_manager: ExpertCacheManager,
        num_experts: int = 128,
        top_k: int = 8,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.cache_manager = cache_manager
        self.num_experts = num_experts
        self.top_k = top_k

        # Keep router on VRAM (tiny)
        self.gate = original_moe.gate

        # Shared expert if model has one
        self.shared_expert = getattr(original_moe, "shared_expert", None)
        self.shared_expert_gate = getattr(original_moe, "shared_expert_gate", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        # Get routing decisions
        router_output = self.gate(hidden_flat)

        if isinstance(router_output, tuple):
            router_logits, router_scores, router_indices = router_output
        else:
            router_logits = router_output
            router_scores, router_indices = torch.topk(
                F.softmax(router_logits, dim=-1), self.top_k, dim=-1
            )

        # Fetch experts (may substitute on miss)
        unique_experts = router_indices.unique().tolist()
        expert_weights = {}
        for expert_id in unique_experts:
            weights, _ = self.cache_manager.get_expert(self.layer_idx, expert_id)
            expert_weights[expert_id] = weights

        # Execute MoE
        output = self._execute_moe(hidden_flat, router_indices, router_scores, expert_weights)

        # Shared expert
        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_flat)
            if self.shared_expert_gate is not None:
                gate_score = torch.sigmoid(self.shared_expert_gate(hidden_flat))
                output = output + gate_score * shared_out
            else:
                output = output + shared_out

        return output.view(batch_size, seq_len, hidden_dim)

    def _execute_moe(
        self,
        hidden: torch.Tensor,
        indices: torch.Tensor,
        scores: torch.Tensor,
        expert_weights: dict[int, ExpertWeights],
    ) -> torch.Tensor:
        num_tokens = hidden.shape[0]
        output = torch.zeros_like(hidden)

        for expert_id, weights in expert_weights.items():
            mask = indices == expert_id
            if not mask.any():
                continue

            token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]
            if len(token_indices) == 0:
                continue

            expert_scores = torch.zeros(num_tokens, device=hidden.device, dtype=hidden.dtype)
            for k in range(self.top_k):
                k_mask = mask[:, k]
                expert_scores[k_mask] = scores[k_mask, k]

            token_hidden = hidden[token_indices]
            expert_out = self._expert_forward(token_hidden, weights)

            weighted_scores = expert_scores[token_indices].unsqueeze(-1)
            output[token_indices] += weighted_scores * expert_out

        return output

    def _expert_forward(self, hidden: torch.Tensor, weights: ExpertWeights) -> torch.Tensor:
        """output = down_proj(silu(gate_proj(x)) * up_proj(x))"""
        gate = F.silu(F.linear(hidden, weights.gate_proj))
        up = F.linear(hidden, weights.up_proj)
        return F.linear(gate * up, weights.down_proj)


class StreamingMoEModel:
    """
    Wrapper that patches a loaded Qwen3 model to use streaming MoE.

    Usage:
        loader = load_model()
        model = StreamingMoEModel(loader)

        with model:
            output = model.generate("Hello!")
    """

    def __init__(self, loader: ModelLoader):
        self.loader = loader
        self.model = loader.model
        self.tokenizer = loader.tokenizer
        self.config = loader.config

        self.cache_manager = ExpertCacheManager(
            vram_experts=loader.vram_experts,
            ram_experts=loader.ram_experts,
            cache_capacity_per_layer=16,
            device="cuda",
        )

        self._original_moe_modules: dict[int, nn.Module] = {}
        self._patched = False

    def start(self):
        """Start cache workers and patch model."""
        self.cache_manager.start()
        self._patch_moe_layers()

    def stop(self):
        """Stop cache workers and restore model."""
        self.cache_manager.stop()
        self._restore_moe_layers()

    def _patch_moe_layers(self):
        if self._patched:
            return

        num_layers = self.config.num_hidden_layers
        num_experts = self.config.num_experts
        top_k = getattr(self.config, "num_experts_per_tok", 8)

        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]
            if not hasattr(layer, "mlp"):
                continue

            original_moe = layer.mlp
            self._original_moe_modules[layer_idx] = original_moe

            streaming_moe = StreamingMoELayer(
                layer_idx=layer_idx,
                original_moe=original_moe,
                cache_manager=self.cache_manager,
                num_experts=num_experts,
                top_k=top_k,
            )

            layer.mlp = streaming_moe

        self._patched = True

    def _restore_moe_layers(self):
        if not self._patched:
            return

        for layer_idx, original_moe in self._original_moe_modules.items():
            self.model.model.layers[layer_idx].mlp = original_moe

        self._original_moe_modules.clear()
        self._patched = False

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }

        # FP8 KV cache (transformers >= 4.45)
        try:
            from transformers import QuantizedCacheConfig

            cache_config = QuantizedCacheConfig(backend="quanto", nbits=8)
            gen_kwargs["cache_implementation"] = "quantized"
            gen_kwargs["cache_config"] = cache_config
        except ImportError:
            pass

        outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input_ids, **kwargs)

    def get_stats(self) -> dict:
        return self.cache_manager.get_stats()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
