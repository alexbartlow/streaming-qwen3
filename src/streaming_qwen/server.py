"""
FastAPI inference server for RAM-streamed Qwen3-30B-A3B.

Full OpenAI-compatible API including:
- /v1/chat/completions with tool/function calling
- /v1/completions (legacy)
- /v1/models
- SSE streaming support

Drop-in replacement for OpenAI API - point opencode or any client at this.
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Union, Literal, Any
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import torch

from .loader import load_model, LayerPlacement
from .model import StreamingMoEModel


# Global model instance
_model: Optional[StreamingMoEModel] = None

MODEL_ID = "qwen3-coder-30b-a3b"


# ============================================================================
# OpenAI-compatible request/response models
# ============================================================================

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None  # For tool messages
    tool_calls: Optional[list[ToolCall]] = None  # For assistant messages
    tool_call_id: Optional[str] = None  # For tool messages


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[Union[str, dict]] = None  # "auto", "none", or {"type": "function", "function": {"name": "..."}}


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# Legacy completions (non-chat)
class CompletionRequest(BaseModel):
    model: str = MODEL_ID
    prompt: Union[str, list[str]]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: None = None
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: ChatCompletionUsage


class StatsResponse(BaseModel):
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    substitutions: int
    prefetches: int
    evictions: int
    vram_layers: list[int]
    ram_layers: list[int]
    vram_allocated_gb: float
    vram_reserved_gb: float


# ============================================================================
# Chat formatting for Qwen3
# ============================================================================

def format_messages_for_qwen3(
    messages: list[ChatMessage],
    tools: Optional[list[ToolDefinition]] = None,
    tokenizer=None,
) -> str:
    """
    Format messages using Qwen3's chat template.

    Qwen3-Coder supports tool calling natively via its chat template.
    """
    # Convert to the format expected by apply_chat_template
    formatted_messages = []

    for msg in messages:
        m = {"role": msg.role, "content": msg.content or ""}

        if msg.role == "assistant" and msg.tool_calls:
            # Format tool calls for Qwen3
            m["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in msg.tool_calls
            ]

        if msg.role == "tool":
            m["tool_call_id"] = msg.tool_call_id
            m["name"] = msg.name

        formatted_messages.append(m)

    # Format tools if provided
    formatted_tools = None
    if tools:
        formatted_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters or {},
                }
            }
            for t in tools
        ]

    # Use tokenizer's chat template
    if tokenizer is not None:
        try:
            prompt = tokenizer.apply_chat_template(
                formatted_messages,
                tools=formatted_tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        except Exception as e:
            # Fallback if chat template fails
            print(f"Chat template error: {e}, using fallback")

    # Fallback: manual formatting
    return _fallback_format(formatted_messages, formatted_tools)


def _fallback_format(messages: list[dict], tools: Optional[list[dict]] = None) -> str:
    """Fallback formatting if chat template unavailable."""
    parts = []

    # Add tools to system message if present
    if tools:
        tool_desc = "You have access to the following tools:\n\n"
        for t in tools:
            func = t["function"]
            tool_desc += f"### {func['name']}\n"
            tool_desc += f"{func.get('description', '')}\n"
            if func.get("parameters"):
                tool_desc += f"Parameters: {json.dumps(func['parameters'], indent=2)}\n"
            tool_desc += "\n"
        tool_desc += "To use a tool, respond with a JSON object in this format:\n"
        tool_desc += '{"name": "tool_name", "arguments": {...}}\n\n'
        parts.append(f"<|im_start|>system\n{tool_desc}<|im_end|>")

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "tool":
            # Format tool response
            name = msg.get("name", "tool")
            parts.append(f"<|im_start|>tool\n[{name}]: {content}<|im_end|>")
        elif role == "assistant" and msg.get("tool_calls"):
            # Format assistant tool call
            tc = msg["tool_calls"][0]
            call_json = json.dumps({
                "name": tc["function"]["name"],
                "arguments": json.loads(tc["function"]["arguments"])
            })
            parts.append(f"<|im_start|>assistant\n{call_json}<|im_end|>")
        else:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def parse_tool_calls(response: str) -> tuple[str, Optional[list[ToolCall]]]:
    """
    Parse tool calls from model response.

    Qwen3 outputs tool calls as JSON. We need to detect and parse them.
    """
    response = response.strip()

    # Try to parse as JSON tool call
    try:
        # Look for JSON object in response
        if response.startswith("{") or response.startswith("["):
            data = json.loads(response)

            # Single tool call
            if isinstance(data, dict) and "name" in data:
                tool_call = ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    function=FunctionCall(
                        name=data["name"],
                        arguments=json.dumps(data.get("arguments", {})),
                    )
                )
                return "", [tool_call]

            # Multiple tool calls
            if isinstance(data, list):
                tool_calls = []
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        tool_calls.append(ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            function=FunctionCall(
                                name=item["name"],
                                arguments=json.dumps(item.get("arguments", {})),
                            )
                        ))
                if tool_calls:
                    return "", tool_calls
    except json.JSONDecodeError:
        pass

    # Check for Qwen3's native tool call format
    # <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    if "<tool_call>" in response:
        import re
        matches = re.findall(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
        if matches:
            tool_calls = []
            for match in matches:
                try:
                    data = json.loads(match.strip())
                    tool_calls.append(ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        function=FunctionCall(
                            name=data["name"],
                            arguments=json.dumps(data.get("arguments", {})),
                        )
                    ))
                except json.JSONDecodeError:
                    continue
            if tool_calls:
                # Remove tool call tags from response
                clean_response = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL).strip()
                return clean_response, tool_calls

    # No tool calls found
    return response, None


# ============================================================================
# Server setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _model

    print("=" * 60)
    print("Loading Qwen3-Coder-30B-A3B with FP8 + RAM streaming")
    print("=" * 60)

    placement = LayerPlacement(
        vram_layers=set(range(0, 6)) | set(range(36, 48)),
        ram_layers=set(range(6, 36)),
        hot_cache_size_per_layer=16,
    )

    loader = load_model(placement=placement)
    _model = StreamingMoEModel(loader)
    _model.start()

    print("=" * 60)
    print(f"Ready! OpenAI-compatible API at http://localhost:8000")
    print(f"Model ID: {MODEL_ID}")
    print("=" * 60)

    yield

    print("Shutting down...")
    if _model is not None:
        _model.stop()
    torch.cuda.empty_cache()


app = FastAPI(
    title="Qwen3-Coder-30B-A3B API",
    description="OpenAI-compatible API with RAM-streamed MoE inference",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================================
# API endpoints
# ============================================================================

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models."""
    return ModelsResponse(
        data=[
            ModelInfo(
                id=MODEL_ID,
                created=int(time.time()),
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions with tool calling support."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Format messages with Qwen3 chat template
    prompt = format_messages_for_qwen3(
        request.messages,
        tools=request.tools,
        tokenizer=_model.tokenizer,
    )

    prompt_tokens = len(_model.tokenizer.encode(prompt))

    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(request, prompt, prompt_tokens),
            media_type="text/event-stream",
        )

    # Non-streaming generation
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(
        None,
        lambda: _model.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens or 2048,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
        )
    )

    # Extract response (remove prompt)
    response_text = output[len(prompt):] if output.startswith(prompt) else output

    # Clean up any trailing special tokens
    for stop_token in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        if stop_token in response_text:
            response_text = response_text.split(stop_token)[0]

    response_text = response_text.strip()

    # Parse for tool calls if tools were provided
    tool_calls = None
    if request.tools:
        response_text, tool_calls = parse_tool_calls(response_text)

    completion_tokens = len(_model.tokenizer.encode(response_text))

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else "stop"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=response_text if response_text else None,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _stream_chat_completion(request: ChatCompletionRequest, prompt: str, prompt_tokens: int):
    """Stream chat completion tokens as SSE events."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Send initial chunk with role
    initial_chunk = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role="assistant"),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Generate and stream tokens
    # Note: This is a simplified streaming implementation
    # Real implementation would need token-by-token generation
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(
        None,
        lambda: _model.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens or 2048,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
        )
    )

    response_text = output[len(prompt):] if output.startswith(prompt) else output
    for stop_token in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        if stop_token in response_text:
            response_text = response_text.split(stop_token)[0]
    response_text = response_text.strip()

    # Stream content in chunks (simulated - real impl would be token-by-token)
    chunk_size = 10
    for i in range(0, len(response_text), chunk_size):
        chunk_text = response_text[i:i + chunk_size]
        chunk = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content=chunk_text),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0.01)  # Small delay for realistic streaming feel

    # Send final chunk
    final_chunk = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """Legacy completions endpoint."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    prompt_tokens = len(_model.tokenizer.encode(prompt))

    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(
        None,
        lambda: _model.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
        )
    )

    completion_text = output[len(prompt):].strip()
    completion_tokens = len(_model.tokenizer.encode(completion_text))

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            CompletionChoice(
                text=completion_text,
                index=0,
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Get cache and memory statistics."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_stats = _model.get_stats()
    cache_stats = model_stats["cache"]

    return StatsResponse(
        cache_hits=cache_stats["hits"],
        cache_misses=cache_stats["misses"],
        cache_hit_rate=cache_stats["hit_rate"],
        substitutions=cache_stats["substitutions"],
        prefetches=cache_stats["prefetches"],
        evictions=cache_stats["evictions"],
        vram_layers=model_stats["vram_layers"],
        ram_layers=model_stats["ram_layers"],
        vram_allocated_gb=torch.cuda.memory_allocated() / 1e9,
        vram_reserved_gb=torch.cuda.memory_reserved() / 1e9,
    )


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy" if _model is not None else "loading",
        "model": MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# Also respond to root for convenience
@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Qwen3-Coder-30B-A3B API",
        "version": "0.1.0",
        "model": MODEL_ID,
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "models": "/v1/models",
            "stats": "/stats",
            "health": "/health",
        },
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the inference server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
