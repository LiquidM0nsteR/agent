# agent/backend/util.py

from __future__ import annotations

import os
import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import HumanMessage

try:
    from langgraph.config import get_stream_writer
except Exception:
    get_stream_writer = None

Intent = Literal["rag", "web_search", "sc_analysis", "chat", "unknown"]
InputKind = Literal["text", "pdf", "h5ad", "markdown", "mixed", "unknown"]
StreamEventCallback = Callable[[dict[str, Any]], None]
_STREAM_EVENT_CALLBACK: ContextVar[StreamEventCallback | None] = ContextVar(
    "agent_stream_event_callback",
    default=None,
)


# =========================
# 1. 全局配置
# =========================

KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "")
SC_OUTPUT_DIR = os.getenv("SC_OUTPUT_DIR", "./outputs/sc_analysis")

UPLOAD_WORKDIR = os.getenv("AGENT_UPLOAD_WORKDIR", "./outputs/uploads")
MAX_GRAPH_STEPS = int(os.getenv("AGENT_MAX_GRAPH_STEPS", "8"))


# =========================
# 2. 文件类型定义
# =========================

TEXT_SUFFIXES = {".txt",".md",".markdown"}
PDF_SUFFIXES = {".pdf"}
H5AD_SUFFIXES = {".h5ad"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
CODE_SUFFIXES = {
    ".py",".cpp",".cc",".cxx",".c",".h",".hpp",".hh",".java",
    ".js",".ts",".tsx",".jsx",".vue",".go",".rs",".sh",".bash",
    ".zsh",".sql",".yaml",".yml",".json",".toml",".ini",".cfg",".r",".R",
}


class UploadedFileInfo(TypedDict, total=False):
    original_path: str
    normalized_path: str
    suffix: str
    kind: InputKind
    converted: bool


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_id(value: str | None, default: str) -> str:
    raw = str(value or "").strip()
    safe = "".join(char for char in raw if char.isalnum() or char in {"-", "_"})
    return safe or default


def safe_filename(filename: str | None) -> str:
    raw = str(filename or "upload.bin").strip()
    safe = "".join(char for char in raw if char.isalnum() or char in {".", "-", "_"})
    return safe or "upload.bin"


def safe_relpath(value: str) -> Path:
    candidate = Path(str(value or "").replace("\\", "/").lstrip("/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Invalid path")
    return candidate


def truncate_text(text: str, limit: int = 96) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:8]}"


def is_local_knowledge_upload_request(text: str) -> bool:
    lowered = str(text or "").lower()
    knowledge_terms = ("本地知识库", "知识库", "local knowledge", "knowledge base")
    action_terms = ("上传", "加入", "添加", "保存", "导入", "copy", "add", "import", "save")
    return any(term in lowered for term in knowledge_terms) and any(term in lowered for term in action_terms)


def is_rag_upload_candidate(saved_file: dict[str, Any]) -> bool:
    kind = str(saved_file.get("kind") or "")
    if kind == "pdf":
        return False
    if kind in {"text", "markdown"}:
        return True
    if kind in {"h5ad", "image"}:
        return False
    suffix = Path(str(saved_file.get("original_name") or saved_file.get("path") or "")).suffix.lower()
    return suffix != ".pdf" and suffix not in H5AD_SUFFIXES and suffix not in IMAGE_SUFFIXES


def has_session_rag_sources(uploads_root: Path) -> bool:
    if not uploads_root.exists():
        return False
    rag_suffixes = PDF_SUFFIXES | TEXT_SUFFIXES
    for path in uploads_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in (rag_suffixes - PDF_SUFFIXES):
            return True
    return False


def build_effective_user_text(raw_text: str, saved_files: list[dict[str, Any]]) -> str:
    text = raw_text.strip()
    if text:
        return text
    if any(item["kind"] == "h5ad" for item in saved_files):
        return "请分析已上传的 h5ad 文件。"
    if any(item["kind"] in {"text", "markdown"} for item in saved_files):
        return "请总结已上传文件的核心内容。"
    if any(item["kind"] == "pdf" for item in saved_files):
        return "已上传 PDF 附件。除非明确要求加入本地知识库，本轮不会自动索引该 PDF。"
    return "请处理已上传的附件。"


# =========================
# 3. LangGraph Streaming 工具
# =========================

@contextmanager
def stream_event_callback(callback: StreamEventCallback | None) -> Iterator[None]:
    token = _STREAM_EVENT_CALLBACK.set(callback)
    try:
        yield
    finally:
        _STREAM_EVENT_CALLBACK.reset(token)


def emit(event: dict[str, Any]) -> None:
    callback = _STREAM_EVENT_CALLBACK.get()
    if callback is not None:
        try:
            callback(event)
        except Exception:
            pass
        return

    if get_stream_writer is None:
        return

    try:
        writer = get_stream_writer()
        writer(event)
    except Exception:
        pass


# =========================
# 4. State / Message 工具
# =========================

def last_user_text(state: dict[str, Any]) -> str:
    if state.get("user_input"):
        return str(state["user_input"])

    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)

    return ""


def safe_text(obj: Any, max_len: int = 6000) -> str:
    text = str(obj)
    if len(text) > max_len:
        return text[:max_len] + "\n...[truncated]"
    return text


def format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)}"


# =========================
# 5. 上传文件归一化
# =========================

def get_file_kind(path: str | Path) -> InputKind:
    suffix = Path(path).suffix

    if suffix in H5AD_SUFFIXES:
        return "h5ad"

    if suffix in PDF_SUFFIXES:
        return "pdf"

    if suffix.lower() in TEXT_SUFFIXES:
        return "markdown" if suffix.lower() in {".md", ".markdown"} else "text"

    if suffix in CODE_SUFFIXES or suffix.lower() in CODE_SUFFIXES:
        return "markdown"

    return "unknown"


def _guess_code_language(path: str | Path) -> str:
    suffix = Path(path).suffix

    mapping = {
        ".py": "python", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".c": "c",
        ".h": "cpp", ".hpp": "cpp", ".java": "java", ".js": "javascript",
        ".ts": "typescript", ".tsx": "tsx", ".jsx": "jsx", ".vue": "vue",
        ".go": "go", ".rs": "rust", ".sh": "bash", ".bash": "bash", ".zsh": "zsh",
        ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".json": "json",
        ".toml": "toml", ".ini": "ini", ".cfg": "ini", ".r": "r", ".R": "r",
    }

    return mapping.get(suffix, mapping.get(suffix.lower(), ""))


def convert_code_file_to_markdown(
    file_path: str | Path,
    output_dir: str | Path = UPLOAD_WORKDIR,
) -> str:
    """
    将 .py / .cpp / .h / .js 等脚本文件转换为 markdown 文件。

    转换格式：

        # Source file: xxx.py

        ```python
        原代码内容
        ```

    返回转换后的 .md 路径。
    """
    src = Path(file_path).expanduser().resolve()

    if not src.exists():
        raise FileNotFoundError(f"上传文件不存在: {src}")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lang = _guess_code_language(src)
    md_name = f"{src.stem}{src.suffix.replace('.', '_')}.md"
    dst = output_dir / md_name

    content = src.read_text(encoding="utf-8", errors="ignore")

    md_content = (
        f"# Source file: {src.name}\n\n"
        f"- Original path: `{src}`\n"
        f"- Converted from: `{src.suffix}`\n\n"
        f"```{lang}\n"
        f"{content}\n"
        f"```\n"
    )

    dst.write_text(md_content, encoding="utf-8")

    return str(dst)


def normalize_uploaded_files(
    file_paths: Optional[list[str]] = None,
    upload_workdir: str = UPLOAD_WORKDIR,
) -> list[UploadedFileInfo]:
    """
    归一化上传文件：
    - .h5ad 保持原路径；
    - .pdf 保持原路径；
    - .txt / .md 保持原路径；
    - .py / .cpp / .h 等代码文件转换为 .md；
    - 其他未知文件暂时保留，但标记为 unknown。
    """
    if not file_paths:
        return []

    normalized: list[UploadedFileInfo] = []

    for raw_path in file_paths:
        path = Path(raw_path).expanduser()
        suffix = path.suffix

        if not path.exists():
            normalized.append(
                {
                    "original_path": str(path),
                    "normalized_path": str(path),
                    "suffix": suffix,
                    "kind": "unknown",
                    "converted": False,
                }
            )
            continue

        if suffix in CODE_SUFFIXES or suffix.lower() in CODE_SUFFIXES:
            md_path = convert_code_file_to_markdown(path, upload_workdir)
            normalized.append(
                {
                    "original_path": str(path),
                    "normalized_path": md_path,
                    "suffix": suffix,
                    "kind": "markdown",
                    "converted": True,
                }
            )
            continue

        kind = get_file_kind(path)

        normalized.append(
            {
                "original_path": str(path),
                "normalized_path": str(path),
                "suffix": suffix,
                "kind": kind,
                "converted": False,
            }
        )

    return normalized


def infer_input_kind_by_files(
    normalized_files: list[UploadedFileInfo],
    user_input: str = "",
) -> InputKind:
    """
    只根据输入形态判断输入类型，不做关键词匹配。

    优先级：
    1. 有 h5ad -> h5ad
    2. 文件类型混合 -> mixed
    3. 有 pdf -> pdf
    4. 有 markdown/text -> markdown/text
    5. 无文件但有文本 -> text
    6. 否则 unknown
    """
    if not normalized_files:
        return "text" if user_input.strip() else "unknown"

    kinds = {f.get("kind", "unknown") for f in normalized_files}

    if "h5ad" in kinds:
        return "h5ad"

    known_kinds = kinds - {"unknown"}

    if len(known_kinds) > 1:
        return "mixed"

    if "pdf" in known_kinds:
        return "pdf"

    if "markdown" in known_kinds:
        return "markdown"

    if "text" in known_kinds:
        return "text"

    return "unknown"


def get_h5ad_files(normalized_files: list[UploadedFileInfo]) -> list[str]:
    return [
        item["normalized_path"]
        for item in normalized_files
        if item.get("kind") == "h5ad"
    ]


def get_rag_files(normalized_files: list[UploadedFileInfo]) -> list[str]:
    """
    RAG 可处理文件：
    - pdf
    - md / markdown
    - txt
    - 由代码文件转换得到的 md
    """
    return [
        item["normalized_path"]
        for item in normalized_files
        if item.get("kind") in {"pdf", "markdown", "text"}
    ]


# =========================
# 6. Observation 工具
# =========================

def latest_observation(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    observations = state.get("observations", [])
    if not observations:
        return None
    return observations[-1]


def tool_already_used(state: dict[str, Any], node_name: str) -> bool:
    observations = state.get("observations", [])
    return any(obs.get("node") == node_name for obs in observations)


# =========================
# 7. Serper 结果格式化
# =========================

def format_serper_results(results: dict[str, Any]) -> str:
    parts: list[str] = []

    answer_box = results.get("answerBox")
    if isinstance(answer_box, dict):
        answer = answer_box.get("answer") or answer_box.get("snippet")
        if answer:
            parts.append(f"[AnswerBox]\n{answer}")

    knowledge_graph = results.get("knowledgeGraph")
    if isinstance(knowledge_graph, dict):
        title = knowledge_graph.get("title", "")
        description = knowledge_graph.get("description", "")

        if title or description:
            parts.append(f"[KnowledgeGraph]\n{title}\n{description}")

    organic = results.get("organic", [])
    if isinstance(organic, list):
        lines = []

        for item in organic[:5]:
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")

            lines.append(
                f"- {title}\n"
                f"  摘要：{snippet}\n"
                f"  链接：{link}"
            )

        if lines:
            parts.append("[Organic]\n" + "\n".join(lines))

    if not parts:
        return safe_text(results)

    return "\n\n".join(parts)


# =========================
# 8. LLM Router 结果解析
# =========================

def parse_router_output(text: str) -> Optional[str]:
    """
    解析 LLM Router 输出。

    允许两种格式：
    1. JSON:
        {"next_node": "RAG"}

    2. 纯节点名:
        RAG

    注意：这里不是意图字符串匹配，而是解析 LLM 已经给出的结构化路由结果。
    """
    if not text:
        return None

    text = text.strip()
    if text in {"RAG", "WebSearch", "FinalNode"}:
        return text

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    node = data.get("next_node")
    if node in {"RAG", "WebSearch", "FinalNode"}:
        return node

    return None
