"""
FP8 Model Loader with layer-aware VRAM/RAM placement.

Key insight: Layers 6-35 have near-zero KL divergence on expert substitution,
so cache misses there are essentially free. Pin sensitive layers (0-5, 36-47)
in VRAM, stream the rest from RAM.
"""

from dataclasses import dataclass, field
from typing import Optional
import gc

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


@dataclass
class LayerPlacement:
    """Configuration for layer placement across VRAM and RAM."""

    # Layers pinned permanently in VRAM (sensitive to expert substitution)
    vram_layers: set[int] = field(default_factory=lambda: set(range(0, 6)) | set(range(36, 48)))

    # Layers stored in CPU RAM, streamed on demand (fungible experts)
    ram_layers: set[int] = field(default_factory=lambda: set(range(6, 36)))

    # Number of experts to keep in VRAM hot cache per RAM layer
    hot_cache_size_per_layer: int = 16

    @property
    def total_layers(self) -> int:
        return len(self.vram_layers) + len(self.ram_layers)

    def is_vram_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.vram_layers

    def is_ram_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.ram_layers


def quantize_to_fp8(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP8 (E4M3) format."""
    if tensor.dtype == torch.float8_e4m3fn:
        return tensor, scale if scale is not None else torch.tensor(1.0)

    amax = tensor.abs().max()
    fp8_max = 448.0  # E4M3 max
    scale = amax / fp8_max if amax > 0 else torch.tensor(1.0, device=tensor.device)

    scaled = tensor / scale
    clamped = scaled.clamp(-fp8_max, fp8_max)
    quantized = clamped.to(torch.float8_e4m3fn)

    return quantized, scale


def dequantize_from_fp8(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 tensor back to BF16."""
    return tensor.to(torch.bfloat16) * scale


@dataclass
class ExpertWeights:
    """Container for a single expert's weights in FP8."""

    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
    gate_scale: torch.Tensor
    up_scale: torch.Tensor
    down_scale: torch.Tensor

    def to_device(self, device: torch.device, dequantize: bool = False) -> "ExpertWeights":
        """Move weights to device. Optionally dequantize to BF16."""
        if dequantize:
            return ExpertWeights(
                gate_proj=dequantize_from_fp8(self.gate_proj.to(device), self.gate_scale.to(device)),
                up_proj=dequantize_from_fp8(self.up_proj.to(device), self.up_scale.to(device)),
                down_proj=dequantize_from_fp8(self.down_proj.to(device), self.down_scale.to(device)),
                gate_scale=self.gate_scale.to(device),
                up_scale=self.up_scale.to(device),
                down_scale=self.down_scale.to(device),
            )
        else:
            # Keep in FP8 - dequantize at inference time
            return ExpertWeights(
                gate_proj=self.gate_proj.to(device),
                up_proj=self.up_proj.to(device),
                down_proj=self.down_proj.to(device),
                gate_scale=self.gate_scale.to(device),
                up_scale=self.up_scale.to(device),
                down_scale=self.down_scale.to(device),
            )

    def dequantize(self) -> "ExpertWeights":
        """Dequantize FP8 weights to BF16 in-place."""
        return ExpertWeights(
            gate_proj=dequantize_from_fp8(self.gate_proj, self.gate_scale),
            up_proj=dequantize_from_fp8(self.up_proj, self.up_scale),
            down_proj=dequantize_from_fp8(self.down_proj, self.down_scale),
            gate_scale=self.gate_scale,
            up_scale=self.up_scale,
            down_scale=self.down_scale,
        )

    @property
    def size_bytes(self) -> int:
        return (
            self.gate_proj.numel()
            + self.up_proj.numel()
            + self.down_proj.numel()
            + (self.gate_scale.numel() + self.up_scale.numel() + self.down_scale.numel()) * 4
        )


@dataclass
class LayerExperts:
    """All experts for a single layer."""

    experts: dict[int, ExpertWeights]
    device: str

    def get_expert(self, expert_id: int) -> ExpertWeights:
        return self.experts[expert_id]

    @property
    def num_experts(self) -> int:
        return len(self.experts)


class ModelLoader:
    """
    Loads Qwen3-Coder-30B-A3B with FP8 quantization and strategic placement.

    Memory layout:
    - VRAM: Embeddings, attention, layer norms, routers, pinned experts
    - RAM: Fungible experts (layers 6-35) in pinned memory for fast DMA
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        placement: Optional[LayerPlacement] = None,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.placement = placement or LayerPlacement()
        self.device = device

        self.vram_experts: dict[int, LayerExperts] = {}
        self.ram_experts: dict[int, LayerExperts] = {}
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.config = None

    def load(self) -> "ModelLoader":
        """Load model with strategic placement."""
        print(f"Loading {self.model_name}...")

        self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        print(f"  Layers: {self.config.num_hidden_layers}")
        print(f"  Experts/layer: {self.config.num_experts}")
        print(f"  VRAM layers: {sorted(self.placement.vram_layers)}")
        print(f"  RAM layers: {sorted(self.placement.ram_layers)}")

        # Use PyTorch's native SDPA (includes flash attention, memory efficient, and math backends)
        # SDPA auto-selects the best backend at runtime based on inputs
        attn_impl = "sdpa"
        print("  Attention: SDPA (PyTorch native, auto-selects flash/memory-efficient/math)")
        self.config._attn_implementation = attn_impl
        self._has_flash_attn = True  # SDPA includes flash attention backend

        # Load to CPU first, then move attention to CUDA
        print("  Loading weights to CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )

        max_context = getattr(self.config, "max_position_embeddings", 131072)
        print(f"  Max context: {max_context:,} tokens")

        print("  Extracting and quantizing experts...")
        self._extract_experts(model)

        print("  Moving attention/embeddings to VRAM...")
        self._setup_base_model(model)

        print("  Done!")
        self._print_memory_usage()
        print(f"  KV cache budget: ~{self._estimate_kv_cache_size(max_context):.1f} GB (FP8)")

        return self

    def _extract_experts(self, model: nn.Module):
        """Extract expert weights, quantize to FP8, place strategically.

        Qwen3 uses fused expert tensors:
        - gate_up_proj: (num_experts, 2 * intermediate_dim, hidden_dim)
        - down_proj: (num_experts, hidden_dim, intermediate_dim)
        """
        num_layers = self.config.num_hidden_layers
        num_experts = self.config.num_experts
        intermediate_size = self.config.moe_intermediate_size

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "experts"):
                continue

            moe_block = layer.mlp
            experts_module = moe_block.experts
            experts_dict = {}

            # Qwen3MoeExperts uses fused tensors: gate_up_proj and down_proj
            # gate_up_proj shape: (num_experts, 2 * intermediate_dim, hidden_dim)
            # down_proj shape: (num_experts, hidden_dim, intermediate_dim)
            gate_up = experts_module.gate_up_proj.data  # (E, 2*I, H)
            down = experts_module.down_proj.data  # (E, H, I)

            for expert_idx in range(num_experts):
                # Split gate_up into gate and up projections
                gate_up_expert = gate_up[expert_idx]  # (2*I, H)
                gate_proj = gate_up_expert[:intermediate_size, :]  # (I, H)
                up_proj = gate_up_expert[intermediate_size:, :]  # (I, H)
                down_proj = down[expert_idx]  # (H, I)

                gate_q, gate_s = quantize_to_fp8(gate_proj)
                up_q, up_s = quantize_to_fp8(up_proj)
                down_q, down_s = quantize_to_fp8(down_proj)

                experts_dict[expert_idx] = ExpertWeights(
                    gate_proj=gate_q,
                    up_proj=up_q,
                    down_proj=down_q,
                    gate_scale=gate_s,
                    up_scale=up_s,
                    down_scale=down_s,
                )

            if self.placement.is_vram_layer(layer_idx):
                # Move experts to VRAM and dequantize to BF16 for fast inference
                # These are pinned layers - worth 2x memory for speed
                for expert_id, weights in experts_dict.items():
                    experts_dict[expert_id] = weights.to_device(torch.device(self.device), dequantize=True)
                self.vram_experts[layer_idx] = LayerExperts(experts_dict, "cuda")
            else:
                # Pin in RAM for fast DMA transfer
                for expert_id, weights in experts_dict.items():
                    experts_dict[expert_id] = ExpertWeights(
                        gate_proj=weights.gate_proj.pin_memory(),
                        up_proj=weights.up_proj.pin_memory(),
                        down_proj=weights.down_proj.pin_memory(),
                        gate_scale=weights.gate_scale.pin_memory(),
                        up_scale=weights.up_scale.pin_memory(),
                        down_scale=weights.down_scale.pin_memory(),
                    )
                self.ram_experts[layer_idx] = LayerExperts(experts_dict, "cpu")

            # Clear original fused weights and free memory
            experts_module.gate_up_proj = None
            experts_module.down_proj = None
            del gate_up, down
            gc.collect()

        gc.collect()
        torch.cuda.empty_cache()

    def _setup_base_model(self, model: nn.Module):
        """Move non-expert components to VRAM."""
        model.model.embed_tokens = model.model.embed_tokens.to(self.device)

        for layer in model.model.layers:
            layer.self_attn = layer.self_attn.to(self.device)
            layer.input_layernorm = layer.input_layernorm.to(self.device)
            layer.post_attention_layernorm = layer.post_attention_layernorm.to(self.device)

            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate = layer.mlp.gate.to(self.device)

        model.model.norm = model.model.norm.to(self.device)
        model.lm_head = model.lm_head.to(self.device)

        # FA2 is now active (configured in config, attention on CUDA)
        if self._has_flash_attn:
            print("  Flash Attention 2: enabled")

        self.model = model

    def _print_memory_usage(self):
        vram_experts_size = sum(
            sum(e.size_bytes for e in layer.experts.values())
            for layer in self.vram_experts.values()
        )
        ram_experts_size = sum(
            sum(e.size_bytes for e in layer.experts.values())
            for layer in self.ram_experts.values()
        )

        vram_allocated = torch.cuda.memory_allocated() / 1e9
        vram_reserved = torch.cuda.memory_reserved() / 1e9

        print(f"\nMemory Usage:")
        print(f"  VRAM allocated: {vram_allocated:.2f} GB")
        print(f"  VRAM reserved: {vram_reserved:.2f} GB")
        print(f"  VRAM experts: {vram_experts_size / 1e9:.2f} GB ({len(self.vram_experts)} layers)")
        print(f"  RAM experts: {ram_experts_size / 1e9:.2f} GB ({len(self.ram_experts)} layers)")

    def _estimate_kv_cache_size(self, max_tokens: int) -> float:
        num_layers = getattr(self.config, "num_hidden_layers", 48)
        num_kv_heads = getattr(self.config, "num_key_value_heads", 8)
        head_dim = getattr(self.config, "head_dim", 128)

        bytes_per_token = 2 * num_layers * num_kv_heads * head_dim
        return (bytes_per_token * max_tokens) / 1e9

    def get_expert(self, layer_idx: int, expert_id: int, device: str = "cuda") -> ExpertWeights:
        if layer_idx in self.vram_experts:
            return self.vram_experts[layer_idx].get_expert(expert_id)
        elif layer_idx in self.ram_experts:
            weights = self.ram_experts[layer_idx].get_expert(expert_id)
            return weights.to_device(torch.device(device))
        else:
            raise ValueError(f"Layer {layer_idx} not found")


def load_model(
    model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    placement: Optional[LayerPlacement] = None,
) -> ModelLoader:
    """Load model with FP8 quantization and strategic layer placement."""
    loader = ModelLoader(model_name, placement)
    return loader.load()
