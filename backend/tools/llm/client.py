from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from functools import lru_cache
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import urlparse

import torch
from transformers import AutoProcessor
from transformers import TextIteratorStreamer
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from ..rag.config import get_config
from qwen_vl_utils import process_vision_info



_LLM_TRACE_BUFFER: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "llm_trace_buffer",
    default=None,
)


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "file", "data"}


def _normalize_image_value(value: str) -> str:
    value = value.strip()
    if _is_url(value):
        return value
    return Path(value).expanduser().resolve().as_uri()


class LocalQwenVLClient:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
            "dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
        }

        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(self.model_path),
                **model_kwargs,
                attn_implementation="flash_attention_2",
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(self.model_path),
                **model_kwargs,
            )
            self.model.to(self.device)

        self.model.eval()

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # 把不同调用方传入的消息统一整理成 chat template 可接受的结构。
        normalized: list[dict[str, Any]] = []

        for message in messages:
            role = str(message.get("role") or "user")
            content = message.get("content", "")

            if isinstance(content, str):
                normalized.append(
                    {
                        "role": role,
                        "content": [{"type": "text", "text": content}],
                    }
                )
                continue

            if not isinstance(content, list):
                normalized.append(
                    {
                        "role": role,
                        "content": [{"type": "text", "text": str(content)}],
                    }
                )
                continue

            normalized_content: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    normalized_content.append(
                        {"type": "text", "text": str(item)}
                    )
                    continue

                item_type = str(item.get("type") or "").strip().lower()

                if item_type == "text":
                    normalized_content.append(
                        {
                            "type": "text",
                            "text": str(item.get("text") or ""),
                        }
                    )
                    continue

                if item_type == "image":
                    image_value = item.get("image") or item.get("url") or item.get("path")
                    if not image_value:
                        continue

                    normalized_item = {
                        "type": "image",
                        "image": _normalize_image_value(str(image_value)),
                    }

                    # 允许可选尺寸控制参数透传
                    for key in ("resized_height", "resized_width", "min_pixels", "max_pixels"):
                        if key in item:
                            normalized_item[key] = item[key]

                    normalized_content.append(normalized_item)
                    continue

                # 先不主动扩视频，但兼容透传
                if item_type == "video":
                    normalized_item = {"type": "video"}
                    if "video" in item:
                        normalized_item["video"] = item["video"]
                    elif "path" in item:
                        normalized_item["video"] = _normalize_image_value(str(item["path"]))
                    normalized_content.append(normalized_item)
                    continue

                # 未知类型，降级成文本
                normalized_content.append(
                    {"type": "text", "text": str(item)}
                )

            normalized.append(
                {
                    "role": role,
                    "content": normalized_content,
                }
            )

        return normalized

    def invoke(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        repetition_penalty: float = 1.05,
        trace_label: str = "local_qwen_invoke",
    ) -> str:
        normalized_messages = self._normalize_messages(messages)

        text = self.processor.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(normalized_messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text_output = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        _append_llm_trace(
            label=trace_label,
            messages=normalized_messages,
            response=text_output,
            model_path=str(self.model_path),
            device=self.device,
        )
        return text_output

    async def ainvoke(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        repetition_penalty: float = 1.05,
        trace_label: str = "local_qwen_ainvoke",
    ) -> str:
        return await asyncio.to_thread(
            self.invoke,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            trace_label=trace_label,
        )


@lru_cache(maxsize=1)
def _get_cached_client(model_path: str) -> LocalQwenVLClient:
    return LocalQwenVLClient(Path(model_path))


def get_local_qwen_client() -> LocalQwenVLClient:
    config = get_config()
    return _get_cached_client(str(config.llm_model_path))


def preload_local_qwen_client() -> dict[str, Any]:
    client = get_local_qwen_client()
    return {
        "model_path": str(client.model_path),
        "device": client.device,
    }


async def local_chat_completion(
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    repetition_penalty: float = 1.05,
    trace_label: str = "local_chat_completion",
) -> dict[str, Any]:
    # 提供一个稳定的小包装，调用方不需要了解底层 client 细节。
    client = get_local_qwen_client()
    text = await client.ainvoke(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        trace_label=trace_label,
    )
    return {
        "message": text,
        "model_path": str(client.model_path),
        "device": client.device,
    }


async def local_chat_completion_stream(
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    repetition_penalty: float = 1.05,
    trace_label: str = "local_chat_completion_stream",
):
    # 使用 transformers streamer 提供逐段输出，供上层直接转成前端流事件。
    client = get_local_qwen_client()
    normalized_messages = client._normalize_messages(messages)
    text = client.processor.apply_chat_template(
        normalized_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(normalized_messages)
    inputs = client.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(client.model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        client.processor,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        skip_prompt=True,
    )
    generation_error: dict[str, Any] = {}

    def _run_generation() -> None:
        try:
            with torch.no_grad():
                client.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    do_sample=temperature > 0,
                    streamer=streamer,
                )
        except Exception as exc:  # pragma: no cover - 生成错误只在运行期出现
            generation_error["error"] = exc
            streamer.on_finalized_text("", stream_end=True)

    worker = Thread(target=_run_generation, daemon=True)
    worker.start()

    collected_chunks: list[str] = []
    iterator = iter(streamer)
    while True:
        chunk = await asyncio.to_thread(next, iterator, None)
        if chunk is None:
            break
        text_chunk = str(chunk)
        if not text_chunk:
            continue
        collected_chunks.append(text_chunk)
        yield text_chunk

    await asyncio.to_thread(worker.join)
    if generation_error.get("error") is not None:
        raise generation_error["error"]

    _append_llm_trace(
        label=trace_label,
        messages=normalized_messages,
        response="".join(collected_chunks).strip(),
        model_path=str(client.model_path),
        device=client.device,
    )


def begin_llm_trace_session() -> Token:
    return _LLM_TRACE_BUFFER.set([])


def end_llm_trace_session(token: Token) -> list[dict[str, Any]]:
    traces = list(_LLM_TRACE_BUFFER.get() or [])
    _LLM_TRACE_BUFFER.reset(token)
    return traces


def _append_llm_trace(
    *,
    label: str,
    messages: list[dict[str, Any]],
    response: str,
    model_path: str,
    device: str,
) -> None:
    buffer = _LLM_TRACE_BUFFER.get()
    if buffer is None:
        return

    buffer.append(
        {
            "label": label,
            "model_path": model_path,
            "device": device,
            "prompt_preview": _build_prompt_preview(messages),
            "response_preview": response[:1200],
        }
    )


def _build_prompt_preview(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content", "")

        if isinstance(content, str):
            text = content.strip()
            if text:
                parts.append(f"[{role}] {text}")
            continue

        if isinstance(content, list):
            text_parts: list[str] = []
            image_count = 0
            video_count = 0

            for item in content:
                if not isinstance(item, dict):
                    text_parts.append(str(item))
                    continue

                item_type = str(item.get("type") or "").strip().lower()
                if item_type == "text":
                    text = str(item.get("text") or "").strip()
                    if text:
                        text_parts.append(text)
                elif item_type == "image":
                    image_count += 1
                elif item_type == "video":
                    video_count += 1

            prefix = []
            if image_count:
                prefix.append(f"{image_count} image(s)")
            if video_count:
                prefix.append(f"{video_count} video(s)")

            merged = " | ".join(prefix + text_parts).strip()
            if merged:
                parts.append(f"[{role}] {merged}")

    return "\n".join(parts)[:1200]
