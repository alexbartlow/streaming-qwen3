#!/usr/bin/env python3
"""
CLI for RAM-streamed Qwen3-Coder-30B-A3B inference.

Usage:
    streaming-qwen                     # Interactive REPL
    streaming-qwen --prompt "Hello"    # Single prompt
    streaming-qwen --server            # Start API server
    streaming-qwen --benchmark         # Run benchmarks
"""

import argparse
import sys
import time

from .loader import load_model, LayerPlacement
from .model import StreamingMoEModel


def parse_args():
    parser = argparse.ArgumentParser(description="RAM-streamed Qwen3-Coder inference")

    parser.add_argument("--prompt", "-p", type=str, help="Single prompt (otherwise REPL)")
    parser.add_argument("--server", action="store_true", help="Run as API server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument(
        "--vram-layers",
        type=str,
        default="0-5,36-47",
        help="Layers to pin in VRAM (default: 0-5,36-47)",
    )
    parser.add_argument(
        "--cache-size", type=int, default=16, help="Experts to cache per RAM layer"
    )

    return parser.parse_args()


def parse_layer_range(spec: str) -> set[int]:
    """Parse '0-5,36-47' into set of ints."""
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return layers


def run_repl(model: StreamingMoEModel, max_tokens: int, temperature: float):
    """Interactive REPL."""
    print("\n" + "=" * 60)
    print("Qwen3-Coder-30B-A3B Streaming Inference")
    print("=" * 60)
    print("Commands: /stats, /clear, /quit")
    print("=" * 60 + "\n")

    conversation = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if user_input.lower() == "/stats":
            stats = model.get_stats()
            cache = stats["cache"]
            print(f"\nCache: {cache['hits']} hits, {cache['misses']} misses")
            print(f"Hit rate: {cache['hit_rate']:.1%}, Substitutions: {cache['substitutions']}\n")
            continue

        if user_input.lower() == "/clear":
            conversation = []
            print("Cleared.\n")
            continue

        conversation.append({"role": "user", "content": user_input})
        prompt = format_conversation(conversation, model.tokenizer)

        print("Assistant: ", end="", flush=True)
        start = time.perf_counter()
        output = model.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
        elapsed = time.perf_counter() - start

        response = extract_response(output, prompt)
        print(response)

        tokens = len(model.tokenizer.encode(response))
        tps = tokens / elapsed if elapsed > 0 else 0
        print(f"\n[{tokens} tok, {tps:.1f} tok/s]\n")

        conversation.append({"role": "assistant", "content": response})


def format_conversation(messages: list[dict], tokenizer) -> str:
    """Format for Qwen3 chat template."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


def extract_response(full_output: str, prompt: str) -> str:
    """Extract new response from output."""
    if full_output.startswith(prompt):
        response = full_output[len(prompt) :].strip()
    elif "Assistant:" in full_output:
        response = full_output.rsplit("Assistant:", 1)[-1].strip()
    else:
        response = full_output.strip()

    for stop in ["<|im_end|>", "<|endoftext|>", "</s>", "User:"]:
        if stop in response:
            response = response.split(stop)[0].strip()

    return response


def run_benchmark(model: StreamingMoEModel, max_tokens: int):
    """Benchmark suite."""
    print("\n" + "=" * 60)
    print("Benchmark")
    print("=" * 60)

    prompts = [
        ("Short", "What is 2 + 2?"),
        ("Medium", "Explain TCP vs UDP in one paragraph."),
        ("Long", "Write a Python function to find the longest common subsequence of two strings."),
    ]

    for name, prompt in prompts:
        print(f"\n{name}...")
        _ = model.generate(prompt, max_new_tokens=10)  # Warmup

        start = time.perf_counter()
        output = model.generate(prompt, max_new_tokens=max_tokens, temperature=0.01)
        elapsed = time.perf_counter() - start

        tokens = len(model.tokenizer.encode(output)) - len(model.tokenizer.encode(prompt))
        tps = tokens / elapsed if elapsed > 0 else 0
        print(f"  {tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")

    stats = model.get_stats()
    print(f"\nCache hit rate: {stats['cache']['hit_rate']:.1%}")


def main():
    args = parse_args()

    if args.server:
        from .server import run_server
        run_server(host=args.host, port=args.port)
        return

    print("Loading model...")
    vram_layers = parse_layer_range(args.vram_layers)
    all_layers = set(range(48))
    ram_layers = all_layers - vram_layers

    placement = LayerPlacement(
        vram_layers=vram_layers,
        ram_layers=ram_layers,
        hot_cache_size_per_layer=args.cache_size,
    )

    loader = load_model(placement=placement)
    model = StreamingMoEModel(loader)
    model.start()

    try:
        if args.benchmark:
            run_benchmark(model, args.max_tokens)
        elif args.prompt:
            output = model.generate(args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature)
            print(output)
        else:
            run_repl(model, args.max_tokens, args.temperature)
    finally:
        model.stop()


if __name__ == "__main__":
    main()
