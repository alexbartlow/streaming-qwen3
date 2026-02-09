"""
Expert Cache with LRU eviction and async RAM->VRAM streaming.

Key insight: layers 6-35 have near-zero KL divergence on expert substitution.
Cache misses are essentially free - just use whatever expert is already hot.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
import threading
import queue

import torch

from .loader import ExpertWeights, LayerExperts


@dataclass
class CacheEntry:
    """A cached expert in VRAM."""

    layer_idx: int
    expert_id: int
    weights: ExpertWeights
    access_count: int = 0
    last_access: int = 0


class ExpertCache:
    """
    LRU cache for expert weights with async prefetch.

    - Fixed-size VRAM budget per fungible layer
    - LRU eviction when budget exceeded
    - Background thread for async RAM->VRAM transfers
    - Fallback to any hot expert on cache miss (exploiting fungibility)
    """

    def __init__(
        self,
        ram_experts: dict[int, LayerExperts],
        capacity_per_layer: int = 16,
        device: str = "cuda",
    ):
        self.ram_experts = ram_experts
        self.capacity_per_layer = capacity_per_layer
        self.device = torch.device(device)

        self.caches: dict[int, OrderedDict[int, CacheEntry]] = {
            layer_idx: OrderedDict() for layer_idx in ram_experts.keys()
        }

        self._access_counter = 0
        self._lock = threading.Lock()

        self._prefetch_queue: queue.Queue[tuple[int, int]] = queue.Queue()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        self.stats = {
            "hits": 0,
            "misses": 0,
            "substitutions": 0,
            "prefetches": 0,
            "evictions": 0,
        }

    def start_prefetch_worker(self):
        """Start background thread for async prefetching."""
        if self._prefetch_thread is not None:
            return

        self._shutdown.clear()
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

    def stop_prefetch_worker(self):
        """Stop background prefetch thread."""
        if self._prefetch_thread is None:
            return

        self._shutdown.set()
        self._prefetch_queue.put((-1, -1))
        self._prefetch_thread.join(timeout=1.0)
        self._prefetch_thread = None

    def _prefetch_worker(self):
        """Background worker that transfers experts from RAM to VRAM."""
        stream = torch.cuda.Stream()

        while not self._shutdown.is_set():
            try:
                layer_idx, expert_id = self._prefetch_queue.get(timeout=0.1)
                if layer_idx < 0:
                    break

                with self._lock:
                    if expert_id in self.caches.get(layer_idx, {}):
                        continue

                with torch.cuda.stream(stream):
                    self._load_expert_to_cache(layer_idx, expert_id)

                self.stats["prefetches"] += 1

            except queue.Empty:
                continue

    def _load_expert_to_cache(self, layer_idx: int, expert_id: int) -> CacheEntry:
        """Load expert from RAM to VRAM cache."""
        if layer_idx not in self.ram_experts:
            raise ValueError(f"Layer {layer_idx} not in RAM storage")

        ram_weights = self.ram_experts[layer_idx].get_expert(expert_id)
        vram_weights = ram_weights.to_device(self.device)

        entry = CacheEntry(
            layer_idx=layer_idx,
            expert_id=expert_id,
            weights=vram_weights,
            access_count=1,
            last_access=self._access_counter,
        )

        with self._lock:
            cache = self.caches[layer_idx]

            while len(cache) >= self.capacity_per_layer:
                evicted_id, evicted_entry = cache.popitem(last=False)
                self.stats["evictions"] += 1
                del evicted_entry

            cache[expert_id] = entry
            cache.move_to_end(expert_id)

        return entry

    def get(
        self,
        layer_idx: int,
        expert_id: int,
        allow_substitution: bool = True,
    ) -> tuple[ExpertWeights, bool, Optional[int]]:
        """
        Get expert weights from cache.

        Returns:
            (weights, was_hit, substituted_id)
        """
        self._access_counter += 1

        with self._lock:
            cache = self.caches.get(layer_idx)
            if cache is None:
                raise ValueError(f"Layer {layer_idx} not a cacheable layer")

            if expert_id in cache:
                entry = cache[expert_id]
                entry.access_count += 1
                entry.last_access = self._access_counter
                cache.move_to_end(expert_id)
                self.stats["hits"] += 1
                return entry.weights, True, None

            self.stats["misses"] += 1

            if allow_substitution and len(cache) > 0:
                sub_id = next(reversed(cache))
                entry = cache[sub_id]
                entry.access_count += 1
                self.stats["substitutions"] += 1
                self._prefetch_queue.put((layer_idx, expert_id))
                return entry.weights, False, sub_id

        entry = self._load_expert_to_cache(layer_idx, expert_id)
        return entry.weights, False, None

    def prefetch(self, layer_idx: int, expert_ids: list[int]):
        """Queue experts for async prefetch."""
        for expert_id in expert_ids:
            if layer_idx in self.caches and expert_id not in self.caches[layer_idx]:
                self._prefetch_queue.put((layer_idx, expert_id))

    def warm_cache(self, layer_idx: int, top_k: int = 8):
        """Warm cache with initial experts."""
        if layer_idx not in self.ram_experts:
            return

        for expert_id in range(min(top_k, 128)):
            if len(self.caches[layer_idx]) >= self.capacity_per_layer:
                break
            try:
                self._load_expert_to_cache(layer_idx, expert_id)
            except Exception:
                pass

    def get_stats(self) -> dict:
        hit_rate = self.stats["hits"] / max(1, self.stats["hits"] + self.stats["misses"])
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_sizes": {layer_idx: len(cache) for layer_idx, cache in self.caches.items()},
        }

    def clear(self):
        with self._lock:
            for cache in self.caches.values():
                cache.clear()
        torch.cuda.empty_cache()


class ExpertCacheManager:
    """
    High-level manager coordinating caching across all layers.

    - VRAM-pinned layers: direct access
    - RAM layers: LRU cache with substitution
    """

    def __init__(
        self,
        vram_experts: dict[int, LayerExperts],
        ram_experts: dict[int, LayerExperts],
        cache_capacity_per_layer: int = 16,
        device: str = "cuda",
    ):
        self.vram_experts = vram_experts
        self.ram_experts = ram_experts
        self.device = device

        self.cache = ExpertCache(
            ram_experts=ram_experts,
            capacity_per_layer=cache_capacity_per_layer,
            device=device,
        )

    def start(self):
        """Start cache workers and warm up."""
        self.cache.start_prefetch_worker()

        for layer_idx in self.ram_experts.keys():
            self.cache.warm_cache(layer_idx, top_k=8)

    def stop(self):
        """Stop cache workers."""
        self.cache.stop_prefetch_worker()

    def get_expert(self, layer_idx: int, expert_id: int) -> tuple[ExpertWeights, bool]:
        """Get expert weights for any layer."""
        if layer_idx in self.vram_experts:
            return self.vram_experts[layer_idx].get_expert(expert_id), True

        weights, was_hit, _ = self.cache.get(layer_idx, expert_id, allow_substitution=True)
        return weights, was_hit

    def prefetch_for_chunk(self, routing_predictions: dict[int, list[int]]):
        """Prefetch experts based on chunk-level routing predictions."""
        for layer_idx, expert_ids in routing_predictions.items():
            if layer_idx in self.ram_experts:
                self.cache.prefetch(layer_idx, expert_ids)

    def get_stats(self) -> dict:
        return {
            "cache": self.cache.get_stats(),
            "vram_layers": list(self.vram_experts.keys()),
            "ram_layers": list(self.ram_experts.keys()),
        }
