# agent/backend/LLM.py

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Sequence, Union

import torch


logger = logging.getLogger(__name__)
DEFAULT_LLM_INSTANCE_COUNT = int(os.getenv("AGENT_LLM_INSTANCE_COUNT", "1"))


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "Qwen2.5-VL-7B-Instruct"
)


@dataclass
class LLMConfig:
    """
    本地 Qwen2.5-VL 模型配置。
    """

    model_path: Union[str, Path] = DEFAULT_MODEL_PATH

    device_map: str = "auto"
    attn_implementation: str = "auto"

    local_files_only: bool = True
    trust_remote_code: bool = True

    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None

    default_system_prompt: str = "你是一个严谨、可靠的本地智能助手。"


@dataclass
class GenerationConfig:
    """
    文本生成参数。
    Agent 场景默认关闭采样，使输出更稳定。
    """

    max_new_tokens: int = 1024
    do_sample: bool = False

    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 50

    repetition_penalty: float = 1.05


@dataclass
class LLMResponse:
    """
    LLM 推理结果。
    """

    text: str
    model_path: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _LLMResource:
    llm: LocalQwenVL
    lock: threading.Lock


class LocalQwenVL:
    """
    本地 Qwen2.5-VL 推理封装。

    该类只负责：
    1. 加载模型；
    2. 构造 Qwen-VL 消息；
    3. 执行文本 / 图文推理；
    4. 返回结构化结果。
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        gen_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.config = config or LLMConfig()
        self.gen_config = gen_config or GenerationConfig()

        self.model = None
        self.processor = None
        self.process_vision_info = None

        self._loaded = False
        self._load_lock = threading.Lock()

    def load(self) -> None:
        """
        懒加载模型。

        注意：
        - 只有第一次真正调用 generate / agenerate 时才加载模型；
        - import LLM.py 不会加载模型；
        - get_llm() 也不会立即加载模型。
        """

        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return

            model_path = Path(self.config.model_path).expanduser().resolve()

            if not model_path.exists():
                raise FileNotFoundError(
                    f"模型路径不存在：{model_path}"
                )

            try:
                from transformers import AutoProcessor
                from transformers import Qwen2_5_VLForConditionalGeneration
            except ImportError as e:
                raise ImportError(
                    "缺少 Qwen2.5-VL 相关 transformers 依赖。"
                    "请安装或升级：pip install -U transformers accelerate"
                ) from e

            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as e:
                raise ImportError(
                    "缺少 qwen-vl-utils。"
                    "请安装：pip install qwen-vl-utils[decord]"
                ) from e

            logger.info("Loading local Qwen2.5-VL model from %s", model_path)

            dtype = torch.float16
            attn_implementation = self._resolve_attention_impl(
                self.config.attn_implementation
            )

            processor_kwargs: Dict[str, Any] = {
                "local_files_only": self.config.local_files_only,
                "trust_remote_code": self.config.trust_remote_code,
            }

            if self.config.min_pixels is not None:
                processor_kwargs["min_pixels"] = self.config.min_pixels

            if self.config.max_pixels is not None:
                processor_kwargs["max_pixels"] = self.config.max_pixels

            self.processor = AutoProcessor.from_pretrained(
                str(model_path),
                **processor_kwargs,
            )

            model_kwargs: Dict[str, Any] = {
                "dtype": dtype,
                "device_map": self.config.device_map,
                "local_files_only": self.config.local_files_only,
                "trust_remote_code": self.config.trust_remote_code,
                "attn_implementation": attn_implementation,
            }

            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(model_path),
                    **model_kwargs,
                )
            except Exception as e:
                if attn_implementation == "flash_attention_2":
                    logger.warning(
                        "flash_attention_2 加载失败，自动回退到 sdpa。错误信息：%s",
                        str(e),
                    )

                    model_kwargs["attn_implementation"] = "sdpa"

                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        str(model_path),
                        **model_kwargs,
                    )
                else:
                    raise

            self.model.eval()
            self.process_vision_info = process_vision_info
            self._loaded = True

            logger.info("Local Qwen2.5-VL model loaded.")

    def generate(
        self,
        prompt: Optional[str] = None,
        images: Optional[Sequence[Union[str, Path, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        gen_config: Optional[GenerationConfig] = None,
        **extra_generate_kwargs: Any,
    ) -> LLMResponse:
        """
        同步推理接口。

        参数
        ----
        prompt: 普通文本输入。
        images: 图片输入，可以是本地路径、file URI、URL 或 PIL Image。
        messages: Qwen chat-template 格式消息。 如果传入 messages，则优先使用 messages。
        system_prompt: 系统提示词。
        gen_config: 本次生成参数。
        extra_generate_kwargs: 额外传递给 model.generate 的参数。
        """

        self.load()

        if messages is None:
            if prompt is None:
                raise ValueError("prompt 和 messages 不能同时为空。")

            messages = self._build_messages(
                prompt=prompt,
                images=images,
                system_prompt=system_prompt,
            )
        else:
            if system_prompt:
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    *messages,
                ]

        generation_config = gen_config or self.gen_config

        assert self.model is not None
        assert self.processor is not None
        assert self.process_vision_info is not None

        start_time = time.time()

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = self.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        input_device = self._get_input_device()
        inputs = inputs.to(input_device)

        generate_kwargs = self._build_generate_kwargs(
            gen_config=generation_config,
            extra_generate_kwargs=extra_generate_kwargs,
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        latency_s = time.time() - start_time

        prompt_tokens = int(inputs.input_ids.shape[-1])
        completion_tokens = int(generated_ids_trimmed[0].numel())

        return LLMResponse(
            text=output_text,
            model_path=str(Path(self.config.model_path).resolve()),
            latency_s=round(latency_s, 4),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def agenerate(
        self,
        prompt: Optional[str] = None,
        images: Optional[Sequence[Union[str, Path, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        gen_config: Optional[GenerationConfig] = None,
        **extra_generate_kwargs: Any,
    ) -> LLMResponse:
        """
        异步推理接口。

        FastAPI 或 LangGraph 异步节点中可以调用。
        """

        return await asyncio.to_thread(
            self.generate,
            prompt=prompt,
            images=images,
            messages=messages,
            system_prompt=system_prompt,
            gen_config=gen_config,
            **extra_generate_kwargs,
        )

    def chat(
        self,
        prompt: str,
        images: Optional[Sequence[Union[str, Path, Any]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        简化同步接口，只返回文本。
        """

        response = self.generate(
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            **kwargs,
        )

        return response.text

    async def achat(
        self,
        prompt: str,
        images: Optional[Sequence[Union[str, Path, Any]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        简化异步接口，只返回文本。
        """

        response = await self.agenerate(
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            **kwargs,
        )

        return response.text

    def _build_messages(
        self,
        prompt: str,
        images: Optional[Sequence[Union[str, Path, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        构造 Qwen2.5-VL messages。
        """

        messages: List[Dict[str, Any]] = []

        final_system_prompt = (
            system_prompt
            if system_prompt is not None
            else self.config.default_system_prompt
        )

        if final_system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": final_system_prompt,
                }
            )

        user_content: List[Dict[str, Any]] = []

        if images:
            for image in images:
                user_content.append(
                    {
                        "type": "image",
                        "image": self._normalize_image_input(image),
                    }
                )

        user_content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        return messages

    @staticmethod
    def _normalize_image_input(image: Union[str, Path, Any]) -> Any:
        """
        规范化图片输入。

        支持：
        1. 本地路径；
        2. file:// URI；
        3. http / https URL；
        4. PIL Image 等对象。
        """

        if isinstance(image, (str, Path)):
            image_str = str(image)

            if image_str.startswith(
                (
                    "http://",
                    "https://",
                    "file://",
                    "data:image",
                )
            ):
                return image_str

            image_path = Path(image_str).expanduser()

            if not image_path.is_absolute():
                image_path = image_path.resolve()

            if not image_path.exists():
                raise FileNotFoundError(f"图片文件不存在：{image_path}")

            return image_path.as_uri()

        return image


    @staticmethod
    def _resolve_attention_impl(attn_implementation: str) -> str:
        """
        解析 attention backend。

        auto 策略：
        1. 优先 flash_attention_2；
        2. 其次 sdpa；
        3. 最后 eager。
        """

        attn_implementation = attn_implementation.lower()

        if attn_implementation != "auto":
            return attn_implementation

        if torch.cuda.is_available() and importlib.util.find_spec("flash_attn"):
            return "flash_attention_2"

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            return "sdpa"

        return "eager"

    def _get_input_device(self) -> torch.device:
        """
        获取模型输入所在设备。
        """

        assert self.model is not None

        try:
            return self.model.device
        except Exception:
            return next(self.model.parameters()).device

    @staticmethod
    def _build_generate_kwargs(
        gen_config: GenerationConfig,
        extra_generate_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        构造 generate 参数。
        """

        kwargs: Dict[str, Any] = {
            "max_new_tokens": gen_config.max_new_tokens,
            "do_sample": gen_config.do_sample,
            "repetition_penalty": gen_config.repetition_penalty,
        }

        if gen_config.do_sample:
            kwargs.update(
                {
                    "temperature": gen_config.temperature,
                    "top_p": gen_config.top_p,
                    "top_k": gen_config.top_k,
                }
            )

        kwargs.update(extra_generate_kwargs)

        return kwargs


_llm_pool: list[_LLMResource] = []
_llm_pool_available: Queue[int] | None = None
_llm_pool_lock = threading.Lock()


def get_llm(
    index: int = 0,
    config: Optional[LLMConfig] = None,
    gen_config: Optional[GenerationConfig] = None,
) -> LocalQwenVL:
    initialize_llm_pool(
        instance_count=DEFAULT_LLM_INSTANCE_COUNT,
        config=config,
        gen_config=gen_config,
    )
    return _llm_pool[index].llm


def initialize_llm_pool(
    instance_count: int = DEFAULT_LLM_INSTANCE_COUNT,
    config: Optional[LLMConfig] = None,
    gen_config: Optional[GenerationConfig] = None,
) -> int:
    global _llm_pool, _llm_pool_available

    if instance_count < 1:
        raise ValueError("LLM instance_count 必须大于 0。")

    with _llm_pool_lock:
        if _llm_pool:
            return len(_llm_pool)

        resources: list[_LLMResource] = []
        available = Queue(maxsize=instance_count)

        for index in range(instance_count):
            llm = LocalQwenVL(
                config=config or LLMConfig(),
                gen_config=gen_config or GenerationConfig(),
            )
            logger.info("Preloading local Qwen2.5-VL instance %s/%s", index + 1, instance_count)
            llm.load()
            resources.append(_LLMResource(llm=llm, lock=threading.Lock()))
            available.put(index)

        _llm_pool = resources
        _llm_pool_available = available
        return len(_llm_pool)


def get_llm_pool_size() -> int:
    return len(_llm_pool)


def reset_llm() -> None:
    global _llm_pool, _llm_pool_available

    with _llm_pool_lock:
        _llm_pool = []
        _llm_pool_available = None


@contextmanager
def acquire_llm(
    config: Optional[LLMConfig] = None,
    gen_config: Optional[GenerationConfig] = None,
):
    initialize_llm_pool(
        instance_count=DEFAULT_LLM_INSTANCE_COUNT,
        config=config,
        gen_config=gen_config,
    )

    assert _llm_pool_available is not None
    index = _llm_pool_available.get()
    resource = _llm_pool[index]
    resource.lock.acquire()
    try:
        yield resource.llm
    finally:
        resource.lock.release()
        _llm_pool_available.put(index)


def chat(
    prompt: str,
    images: Optional[Sequence[Union[str, Path, Any]]] = None,
    system_prompt: Optional[str] = None,
    **kwargs: Any,
) -> str:
    with acquire_llm() as llm:
        return llm.chat(
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            **kwargs,
        )


async def achat(
    prompt: str,
    images: Optional[Sequence[Union[str, Path, Any]]] = None,
    system_prompt: Optional[str] = None,
    **kwargs: Any,
) -> str:
    return await asyncio.to_thread(
        chat,
        prompt=prompt,
        images=images,
        system_prompt=system_prompt,
        **kwargs,
    )
