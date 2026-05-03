# backend/tools/LLM.py
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from openai import OpenAI

from .. import config as C


DEFAULT_LLM_INSTANCE_COUNT = C.AGENT_LLM_INSTANCE_COUNT
DEFAULT_MODEL_PATH = C.VLLM_MODEL_PATH


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = C.VLLM_MAX_NEW_TOKENS
    temperature: float = C.VLLM_TEMPERATURE
    top_p: float = C.VLLM_TOP_P
    top_k: int = C.VLLM_TOP_K
    repetition_penalty: float = C.VLLM_REPETITION_PENALTY
    do_sample: bool = C.VLLM_DO_SAMPLE


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    usage: dict[str, Any]
    finish_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _client() -> OpenAI:
    return OpenAI(base_url=C.VLLM_BASE_URL, api_key=C.VLLM_API_KEY, timeout=C.VLLM_TIMEOUT)


def _normalize_image(image: str | Path | Any) -> dict[str, Any]:
    if not isinstance(image, (str, Path)):
        return image
    value = str(image)
    if value.startswith(("http://", "https://", "data:image", "file://")):
        return {"type": "image_url", "image_url": {"url": value}}
    return {"type": "image_url", "image_url": {"url": Path(value).expanduser().resolve().as_uri()}}


def build_messages(
    prompt: str,
    images: Sequence[str | Path | Any] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if images:
        content: list[Any] = [_normalize_image(image) for image in images]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def _generation_kwargs(config: GenerationConfig | None, extra: dict[str, Any]) -> dict[str, Any]:
    cfg = config or GenerationConfig()
    extra_body = {"top_k": cfg.top_k, "repetition_penalty": cfg.repetition_penalty}
    extra_body.update(dict(extra.pop("extra_body", {}) or {}))
    kwargs: dict[str, Any] = {
        "max_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature if cfg.do_sample else 0,
        "top_p": cfg.top_p,
        "extra_body": extra_body,
    }
    kwargs.update(extra)
    return kwargs


def generate(
    prompt: str | None = None,
    *,
    messages: list[dict[str, Any]] | None = None,
    images: Sequence[str | Path | Any] | None = None,
    system_prompt: str | None = None,
    gen_config: GenerationConfig | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LLMResponse:
    if messages is None:
        if prompt is None:
            raise ValueError("prompt 和 messages 不能同时为空。")
        messages = build_messages(prompt, images=images, system_prompt=system_prompt)

    completion = _client().chat.completions.create(
        model=model or C.VLLM_MODEL,
        messages=messages,
        stream=False,
        **_generation_kwargs(gen_config, kwargs),
    )
    choice = completion.choices[0]
    usage = completion.usage.model_dump() if completion.usage else {}
    return LLMResponse(
        text=(choice.message.content or "").strip(),
        model=completion.model,
        usage=usage,
        finish_reason=str(choice.finish_reason or ""),
    )


def stream_generate(
    prompt: str | None = None,
    *,
    messages: list[dict[str, Any]] | None = None,
    images: Sequence[str | Path | Any] | None = None,
    system_prompt: str | None = None,
    gen_config: GenerationConfig | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> Iterable[str]:
    if messages is None:
        if prompt is None:
            raise ValueError("prompt 和 messages 不能同时为空。")
        messages = build_messages(prompt, images=images, system_prompt=system_prompt)

    stream = _client().chat.completions.create(
        model=model or C.VLLM_MODEL,
        messages=messages,
        stream=True,
        **_generation_kwargs(gen_config, kwargs),
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta


def chat(
    prompt: str,
    images: Sequence[str | Path | Any] | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> str:
    return generate(prompt=prompt, images=images, system_prompt=system_prompt, **kwargs).text


def chat_stream(
    prompt: str,
    images: Sequence[str | Path | Any] | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> Iterable[str]:
    yield from stream_generate(prompt=prompt, images=images, system_prompt=system_prompt, **kwargs)


async def achat(
    prompt: str,
    images: Sequence[str | Path | Any] | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> str:
    return await asyncio.to_thread(chat, prompt, images=images, system_prompt=system_prompt, **kwargs)


def initialize_llm_pool(instance_count: int = DEFAULT_LLM_INSTANCE_COUNT, **_: Any) -> int:
    del instance_count
    models = _client().models.list()
    names = {item.id for item in models.data}
    if C.VLLM_MODEL not in names:
        raise RuntimeError(f"vLLM 服务中未找到模型 {C.VLLM_MODEL!r}，当前可用模型：{sorted(names)}")
    return 1


def get_llm_pool_size() -> int:
    return 1


def reset_llm() -> None:
    return None
