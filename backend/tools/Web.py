# WEB.py
# -*- coding: utf-8 -*-

"""
WEB.py

职责：
1. 读取环境变量 SERPER_API_KEY；
2. 调用 Serper Google Search API；
3. 支持普通搜索、新闻、图片、学术、视频、地点等检索类型；
4. 将 Serper 原始 JSON 解析为统一结构；
5. 返回适合 Agent / LLM 使用的格式化检索结果。

注意：
- 本文件只实现 Web 检索功能，不负责意图识别；
- 是否调用该工具由上层 Agent / LangGraph / Function Calling 决定；
- 不要在代码中硬编码 API Key。
"""

from __future__ import annotations

import os
import re
import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal

try:
    import requests
except ImportError as exc:
    raise ImportError(
        "WEB.py requires `requests`. Install it with: pip install requests"
    ) from exc


logger = logging.getLogger(__name__)


SearchType = Literal[
    "search",
    "news",
    "images",
    "videos",
    "places",
    "maps",
    "shopping",
    "scholar",
    "patents",
    "autocomplete",
]


class WebSearchError(RuntimeError):
    """Raised when web search fails."""


@dataclass
class WebResult:
    title: str
    link: str = ""
    snippet: str = ""
    source: str = ""
    date: str = ""
    position: int = 0
    result_type: str = "organic"
    image_url: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WebSearchOutput:
    query: str
    search_type: str
    answer: str = ""
    results: List[WebResult] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        data = {
            "query": self.query,
            "search_type": self.search_type,
            "answer": self.answer,
            "results": [r.to_dict() for r in self.results],
        }
        if include_raw:
            data["raw"] = self.raw
        return data

    def format_for_llm(self, max_results: int = 8) -> str:
        """
        将搜索结果格式化为适合 LLM 阅读的上下文。
        """
        lines: List[str] = []
        lines.append(f"Query: {self.query}")
        lines.append(f"Search Type: {self.search_type}")

        if self.answer:
            lines.append("")
            lines.append("Direct Answer / Summary:")
            lines.append(self.answer.strip())

        if self.results:
            lines.append("")
            lines.append("Search Results:")

        for idx, item in enumerate(self.results[:max_results], start=1):
            lines.append(f"[{idx}] {item.title}")

            if item.link:
                lines.append(f"URL: {item.link}")

            meta = []
            if item.source:
                meta.append(f"source={item.source}")
            if item.date:
                meta.append(f"date={item.date}")
            if item.result_type:
                meta.append(f"type={item.result_type}")

            if meta:
                lines.append("Meta: " + "; ".join(meta))

            if item.snippet:
                lines.append(f"Snippet: {item.snippet}")

            if item.image_url:
                lines.append(f"Image: {item.image_url}")

            lines.append("")

        return "\n".join(lines).strip()


class SerperClient:
    """
    Serper API 客户端。

    Serper API endpoint 基本形式：
        https://google.serper.dev/search
        https://google.serper.dev/news
        https://google.serper.dev/images
        ...

    认证方式：
        Header: X-API-KEY
    """

    BASE_URL = "https://google.serper.dev"

    SUPPORTED_TYPES = {
        "search",
        "news",
        "images",
        "videos",
        "places",
        "maps",
        "shopping",
        "scholar",
        "patents",
        "autocomplete",
    }

    RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 20,
        max_retries: int = 2,
        default_gl: str = "us",
        default_hl: str = "en",
    ) -> None:
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_gl = default_gl
        self.default_hl = default_hl

        if not self.api_key:
            raise WebSearchError(
                "SERPER_API_KEY is not set. Please run: "
                'export SERPER_API_KEY="your-serper-api-key"'
            )

    def search(
        self,
        query: str,
        search_type: SearchType = "search",
        num: int = 8,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        location: Optional[str] = None,
        tbs: Optional[str] = None,
        page: Optional[int] = None,
        include_raw: bool = True,
    ) -> WebSearchOutput:
        """
        执行一次 Serper 搜索。

        Args:
            query: 搜索问题。
            search_type: 搜索类型，例如 search/news/images/scholar。
            num: 返回结果数量，建议 3-10。
            gl: Google country，例如 us、cn、tw。
            hl: interface language，例如 en、zh-cn、zh-tw。
            location: 具体地理位置，例如 "Taipei, Taiwan"。
            tbs: Google 时间过滤参数，例如：
                qdr:h   过去一小时
                qdr:d   过去一天
                qdr:w   过去一周
                qdr:m   过去一月
                qdr:y   过去一年
            page: 页码。
            include_raw: 是否保留 Serper 原始 JSON。

        Returns:
            WebSearchOutput
        """
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty.")

        if search_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported search_type: {search_type}. "
                f"Supported types: {sorted(self.SUPPORTED_TYPES)}"
            )

        num = max(1, min(int(num), 20))

        payload: Dict[str, Any] = {
            "q": query,
            "num": num,
            "gl": gl or self.default_gl,
            "hl": hl or self.default_hl,
        }

        if location:
            payload["location"] = location

        if tbs:
            payload["tbs"] = tbs

        if page is not None:
            payload["page"] = page

        raw = self._post(search_type=search_type, payload=payload)
        answer = self._extract_answer(raw)
        results = self._parse_results(raw, search_type=search_type, limit=num)

        return WebSearchOutput(
            query=query,
            search_type=search_type,
            answer=answer,
            results=results,
            raw=raw if include_raw else {},
        )

    def _post(self, search_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{search_type}"

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code < 400:
                    try:
                        return response.json()
                    except json.JSONDecodeError as exc:
                        raise WebSearchError(
                            f"Serper returned non-JSON response: {response.text[:300]}"
                        ) from exc

                if (
                    response.status_code in self.RETRY_STATUS_CODES
                    and attempt < self.max_retries
                ):
                    sleep_seconds = 1.5 * (attempt + 1)
                    logger.warning(
                        "Serper temporary error: status=%s, retry=%s/%s",
                        response.status_code,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(sleep_seconds)
                    continue

                raise WebSearchError(
                    f"Serper request failed. "
                    f"status={response.status_code}, body={response.text[:500]}"
                )

            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    sleep_seconds = 1.5 * (attempt + 1)
                    logger.warning(
                        "Serper network error: %s, retry=%s/%s",
                        exc,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(sleep_seconds)
                    continue

        raise WebSearchError(f"Serper request failed: {last_error}")

    @staticmethod
    def _extract_answer(raw: Dict[str, Any]) -> str:
        """
        提取 answerBox / knowledgeGraph 等直接答案。
        """
        parts: List[str] = []

        answer_box = raw.get("answerBox")
        if isinstance(answer_box, dict):
            title = answer_box.get("title") or ""
            answer = answer_box.get("answer") or ""
            snippet = answer_box.get("snippet") or ""

            if title:
                parts.append(str(title))
            if answer:
                parts.append(str(answer))
            if snippet:
                parts.append(str(snippet))

        knowledge_graph = raw.get("knowledgeGraph")
        if isinstance(knowledge_graph, dict):
            kg_title = knowledge_graph.get("title") or ""
            kg_type = knowledge_graph.get("type") or ""
            kg_desc = knowledge_graph.get("description") or ""

            kg_line = " - ".join(x for x in [kg_title, kg_type] if x)
            if kg_line:
                parts.append(kg_line)
            if kg_desc:
                parts.append(str(kg_desc))

        return "\n".join(parts).strip()

    def _parse_results(
        self,
        raw: Dict[str, Any],
        search_type: str,
        limit: int,
    ) -> List[WebResult]:
        """
        将不同类型的 Serper 返回解析成统一 WebResult。
        """
        result_keys = self._candidate_result_keys(search_type)

        parsed: List[WebResult] = []

        for key in result_keys:
            items = raw.get(key)
            if not isinstance(items, list):
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue

                result = self._parse_single_item(item, result_type=key)
                if result is not None:
                    parsed.append(result)

                if len(parsed) >= limit:
                    return parsed

        return parsed[:limit]

    @staticmethod
    def _candidate_result_keys(search_type: str) -> List[str]:
        """
        Serper 不同 endpoint 的结果字段名略有不同。
        这里做宽松兼容。
        """
        mapping = {
            "search": ["organic", "peopleAlsoAsk", "relatedSearches"],
            "news": ["news"],
            "images": ["images"],
            "videos": ["videos"],
            "places": ["places"],
            "maps": ["places"],
            "shopping": ["shopping"],
            "scholar": ["organic", "scholar"],
            "patents": ["organic", "patents"],
            "autocomplete": ["suggestions"],
        }

        return mapping.get(search_type, ["organic"])

    @staticmethod
    def _parse_single_item(
        item: Dict[str, Any],
        result_type: str,
    ) -> Optional[WebResult]:
        title = (
            item.get("title")
            or item.get("name")
            or item.get("value")
            or item.get("question")
            or ""
        )

        title = str(title).strip()
        if not title:
            return None

        link = (
            item.get("link")
            or item.get("url")
            or item.get("website")
            or item.get("sourceLink")
            or ""
        )

        snippet = (
            item.get("snippet")
            or item.get("description")
            or item.get("answer")
            or ""
        )

        source = (
            item.get("source")
            or item.get("domain")
            or item.get("publisher")
            or ""
        )

        date = (
            item.get("date")
            or item.get("publishedDate")
            or item.get("year")
            or ""
        )

        image_url = (
            item.get("imageUrl")
            or item.get("thumbnailUrl")
            or item.get("thumbnail")
            or ""
        )

        position = item.get("position") or 0
        try:
            position = int(position)
        except Exception:
            position = 0

        # places / maps 类结果补充地址、评分等信息
        if result_type in {"places", "maps"}:
            extra_snippets = []
            for key in ["address", "category", "phoneNumber"]:
                if item.get(key):
                    extra_snippets.append(f"{key}: {item[key]}")

            if item.get("rating"):
                extra_snippets.append(f"rating: {item.get('rating')}")

            if item.get("ratingCount"):
                extra_snippets.append(f"ratingCount: {item.get('ratingCount')}")

            if extra_snippets:
                snippet = "；".join([snippet] + extra_snippets if snippet else extra_snippets)

        # shopping 类结果补充价格
        if result_type == "shopping":
            extra_snippets = []
            for key in ["price", "delivery", "rating", "ratingCount"]:
                if item.get(key):
                    extra_snippets.append(f"{key}: {item[key]}")
            if extra_snippets:
                snippet = "；".join([snippet] + extra_snippets if snippet else extra_snippets)

        ignored_keys = {
            "title",
            "name",
            "value",
            "question",
            "link",
            "url",
            "website",
            "sourceLink",
            "snippet",
            "description",
            "answer",
            "source",
            "domain",
            "publisher",
            "date",
            "publishedDate",
            "year",
            "imageUrl",
            "thumbnailUrl",
            "thumbnail",
            "position",
        }

        extra = {k: v for k, v in item.items() if k not in ignored_keys}

        return WebResult(
            title=title,
            link=str(link).strip(),
            snippet=str(snippet).strip(),
            source=str(source).strip(),
            date=str(date).strip(),
            position=position,
            result_type=result_type,
            image_url=str(image_url).strip(),
            extra=extra,
        )


def freshness_to_tbs(freshness: Optional[str]) -> Optional[str]:
    """
    将自然参数转换为 Google tbs 时间过滤参数。

    Examples:
        "hour"  -> "qdr:h"
        "day"   -> "qdr:d"
        "week"  -> "qdr:w"
        "month" -> "qdr:m"
        "year"  -> "qdr:y"
        "qdr:d3" -> "qdr:d3"
    """
    if not freshness:
        return None

    freshness = freshness.strip().lower()

    if freshness.startswith("qdr:"):
        return freshness

    mapping = {
        "h": "qdr:h",
        "hour": "qdr:h",
        "past_hour": "qdr:h",

        "d": "qdr:d",
        "day": "qdr:d",
        "today": "qdr:d",
        "past_day": "qdr:d",

        "w": "qdr:w",
        "week": "qdr:w",
        "past_week": "qdr:w",

        "m": "qdr:m",
        "month": "qdr:m",
        "past_month": "qdr:m",

        "y": "qdr:y",
        "year": "qdr:y",
        "past_year": "qdr:y",
    }

    return mapping.get(freshness)


_CLIENT: Optional[SerperClient] = None


def get_serper_client() -> SerperClient:
    """
    获取全局 SerperClient，避免每次工具调用都重复初始化。
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = SerperClient()
    return _CLIENT


def web_search(
    query: str,
    k: int = 6,
    search_type: SearchType = "search",
    gl: str = "us",
    hl: str = "en",
    location: Optional[str] = None,
    freshness: Optional[str] = None,
    return_json: bool = False,
    include_raw: bool = False,
) -> str | Dict[str, Any]:
    """
    Agent 推荐直接调用的函数。

    Args:
        query: 用户查询。
        k: 返回结果数量。
        search_type: search/news/images/videos/places/scholar 等。
        gl: 国家区域，默认 us，英文信息源更稳定。
        hl: 语言，默认 en。
        location: 地理位置。
        freshness: 时间过滤，例如 day/week/month/year/qdr:d3。
        return_json: True 时返回 dict；False 时返回格式化字符串。
        include_raw: return_json=True 时是否包含 Serper 原始 JSON。

    Returns:
        str 或 dict
    """
    client = get_serper_client()

    output = client.search(
        query=query,
        search_type=search_type,
        num=k,
        gl=gl,
        hl=hl,
        location=location,
        tbs=freshness_to_tbs(freshness),
        include_raw=include_raw,
    )

    if return_json:
        return output.to_dict(include_raw=include_raw)

    return output.format_for_llm(max_results=k)


def web_news_search(
    query: str,
    k: int = 6,
    freshness: Optional[str] = "week",
    gl: str = "us",
    hl: str = "en",
) -> str:
    """
    新闻搜索快捷函数。
    """
    return str(
        web_search(
            query=query,
            k=k,
            search_type="news",
            gl=gl,
            hl=hl,
            freshness=freshness,
            return_json=False,
        )
    )


def web_image_search(
    query: str,
    k: int = 6,
    gl: str = "us",
    hl: str = "en",
) -> Dict[str, Any]:
    """
    图片搜索快捷函数，通常返回 JSON 更方便前端展示。
    """
    return dict(
        web_search(
            query=query,
            k=k,
            search_type="images",
            gl=gl,
            hl=hl,
            return_json=True,
            include_raw=False,
        )
    )


def read_web_page(
    url: str,
    max_chars: int = 8000,
    timeout: int = 15,
) -> Dict[str, Any]:
    """
    轻量级网页正文读取函数。

    说明：
    - 这是辅助能力，不依赖 Serper；
    - 很多网站有反爬、JS 渲染、登录墙，因此失败是正常情况；
    - 对 RAG 来说，优先使用 search snippet，只有需要更长上下文时再读取网页正文。
    """
    url = url.strip()
    if not url:
        raise ValueError("url cannot be empty.")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WebSearchError(f"Failed to fetch URL: {url}. Error: {exc}") from exc

    content_type = response.headers.get("Content-Type", "")

    text = response.text

    # 优先使用 BeautifulSoup；如果环境没有 bs4，则使用正则粗略清洗。
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        paragraphs = []
        for p in soup.find_all(["p", "article", "section", "h1", "h2", "h3"]):
            t = p.get_text(" ", strip=True)
            if t:
                paragraphs.append(t)

        cleaned = "\n".join(paragraphs)

    except ImportError:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.I | re.S)
        title = title_match.group(1).strip() if title_match else ""

        cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
        cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
        cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    cleaned = cleaned[:max_chars]

    return {
        "url": url,
        "content_type": content_type,
        "title": title,
        "text": cleaned,
    }


# 可选：给 LangChain / LangGraph Function Calling 使用的工具函数。
# 如果你不想引入 langchain_core，可以不使用这一段。
def get_langchain_tools():
    """
    返回 LangChain-compatible tools。

    用法：
        tools = get_langchain_tools()
        agent = create_agent(model, tools)
    """
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise ImportError(
            "get_langchain_tools requires langchain-core. "
            "Install it with: pip install langchain-core"
        ) from exc

    @tool
    def search_web_tool(query: str) -> str:
        """
        Search the web for recent or external information.
        Use this when the answer may require up-to-date knowledge.
        """
        return str(web_search(query=query, k=6, search_type="search"))

    @tool
    def search_news_tool(query: str) -> str:
        """
        Search recent news from the web.
        Use this for current events, latest updates, policy changes, product updates, or recent papers.
        """
        return web_news_search(query=query, k=6, freshness="week")

    @tool
    def read_web_page_tool(url: str) -> str:
        """
        Read the textual content of a web page by URL.
        Use this after search_web_tool if the snippet is not enough.
        """
        data = read_web_page(url)
        return (
            f"URL: {data['url']}\n"
            f"Title: {data['title']}\n"
            f"Content-Type: {data['content_type']}\n\n"
            f"{data['text']}"
        )

    return [search_web_tool, search_news_tool, read_web_page_tool]