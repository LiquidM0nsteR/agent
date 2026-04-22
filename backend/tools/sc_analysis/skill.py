from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib
matplotlib.use("Agg")
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ...prompts import (
    build_single_cell_param_messages,
    build_single_cell_report_context,
)
from ..llm import local_chat_completion
from .core import run_analysis_to_dir


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "scgpt_cls"
DEFAULT_N_HVG = 1200
DEFAULT_GENE_CORR_THR = 0.3
DEFAULT_GENE_CORR_TOPK = 10
DEFAULT_STR_BATCH = "str_batch"


def _safe_user_id(user_id: str | None) -> str:
    raw = (user_id or "").strip()
    sanitized = "".join(char for char in raw if char.isalnum() or char in {"-", "_"})
    return sanitized or "anonymous"


def _session_root(user_id: str, session_id: str) -> Path:
    return BASE_DIR / "data" / "users" / _safe_user_id(user_id) / "sessions" / session_id


@dataclass(slots=True)
class SingleCellAnalysisParams:
    need_batch_correction: bool = False
    need_gene_corr: bool = False
    gene_list: list[str] | None = None
    n_hvg: int = DEFAULT_N_HVG
    gene_corr_thr: float = DEFAULT_GENE_CORR_THR
    gene_corr_topk: int = DEFAULT_GENE_CORR_TOPK
    str_batch: str = DEFAULT_STR_BATCH
    model_dir_cls: str = str(DEFAULT_MODEL_DIR)

    def normalized(self) -> "SingleCellAnalysisParams":
        genes = [gene.strip() for gene in (self.gene_list or []) if gene and gene.strip()]
        unique_genes: list[str] = []
        seen: set[str] = set()
        for gene in genes:
            gene_key = gene.upper()
            if gene_key in seen:
                continue
            seen.add(gene_key)
            unique_genes.append(gene)
        return SingleCellAnalysisParams(
            need_batch_correction=bool(self.need_batch_correction),
            need_gene_corr=bool(self.need_gene_corr),
            gene_list=unique_genes,
            n_hvg=max(100, int(self.n_hvg or DEFAULT_N_HVG)),
            gene_corr_thr=min(
                1.0,
                max(0.0, float(self.gene_corr_thr or DEFAULT_GENE_CORR_THR)),
            ),
            gene_corr_topk=max(1, int(self.gene_corr_topk or DEFAULT_GENE_CORR_TOPK)),
            str_batch=(self.str_batch or DEFAULT_STR_BATCH).strip() or DEFAULT_STR_BATCH,
            model_dir_cls=self.model_dir_cls or str(DEFAULT_MODEL_DIR),
        )


def _strip_code_fence(payload: str) -> str:
    text = payload.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(payload: str) -> dict[str, Any] | None:
    text = _strip_code_fence(payload)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_bool_from_text(
    text: str,
    positive_tokens: tuple[str, ...],
    negative_tokens: tuple[str, ...],
) -> bool | None:
    lower = text.lower()
    if any(token.lower() in lower for token in negative_tokens):
        return False
    if any(token.lower() in lower for token in positive_tokens):
        return True
    return None


def _extract_gene_list_from_text(text: str) -> list[str]:
    bracket_match = re.search(
        r"(?:gene_list|基因列表|genes?)\s*[:：]?\s*(\[[^\]]+\])",
        text,
        flags=re.IGNORECASE,
    )
    if bracket_match:
        try:
            parsed = json.loads(bracket_match.group(1).replace("'", '"'))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass

    suffix_match = re.search(
        r"(?:gene_list|基因列表|genes?)\s*[:：]?\s*([A-Za-z0-9_,\-\s]+)",
        text,
        flags=re.IGNORECASE,
    )
    if suffix_match:
        return [
            chunk.strip()
            for chunk in re.split(r"[,，\s]+", suffix_match.group(1).strip())
            if chunk.strip()
        ]
    return []


def _extract_number(
    pattern: str,
    text: str,
    cast: type[int] | type[float],
    default: int | float,
) -> int | float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return default
    try:
        return cast(match.group(1))
    except (TypeError, ValueError):
        return default


def _heuristic_parse_params(user_text: str) -> SingleCellAnalysisParams:
    need_batch_correction = _parse_bool_from_text(
        user_text,
        ("batch correction", "batch effect", "批次矫正", "去批次", "消除批次效应"),
        ("no batch correction", "不要批次矫正", "无需批次矫正", "不做批次矫正"),
    )
    need_gene_corr = _parse_bool_from_text(
        user_text,
        ("gene corr", "gene correlation", "基因相关", "相关网络", "共表达"),
        ("no gene corr", "no gene correlation", "不要基因相关", "不做基因相关"),
    )
    str_batch_match = re.search(
        r"(?:str_batch|batch column|批次列)\s*[:：]?\s*([A-Za-z0-9_.-]+)",
        user_text,
        flags=re.IGNORECASE,
    )
    return SingleCellAnalysisParams(
        need_batch_correction=bool(need_batch_correction),
        need_gene_corr=bool(need_gene_corr),
        gene_list=_extract_gene_list_from_text(user_text),
        n_hvg=int(
            _extract_number(
                r"(?:n_hvg|hvg)\s*[:：]?\s*(\d+)",
                user_text,
                int,
                DEFAULT_N_HVG,
            )
        ),
        gene_corr_thr=float(
            _extract_number(
                r"(?:gene_corr_thr|corr_threshold|相关阈值)\s*[:：]?\s*([0-9]*\.?[0-9]+)",
                user_text,
                float,
                DEFAULT_GENE_CORR_THR,
            )
        ),
        gene_corr_topk=int(
            _extract_number(
                r"(?:gene_corr_topk|corr_topk|topk)\s*[:：]?\s*(\d+)",
                user_text,
                int,
                DEFAULT_GENE_CORR_TOPK,
            )
        ),
        str_batch=str_batch_match.group(1) if str_batch_match else DEFAULT_STR_BATCH,
        model_dir_cls=str(DEFAULT_MODEL_DIR),
    ).normalized()


async def _llm_parse_params(user_text: str) -> SingleCellAnalysisParams | None:
    messages = build_single_cell_param_messages(user_text)
    try:
        local_result = await local_chat_completion(
            messages,
            max_new_tokens=256,
            temperature=0.0,
            trace_label="single_cell_param_parse",
        )
    except Exception:
        return None

    content = str(local_result.get("message") or "")
    parsed = _extract_json_object(content)
    if not parsed:
        return None
    try:
        return SingleCellAnalysisParams(
            need_batch_correction=bool(parsed.get("need_batch_correction", False)),
            need_gene_corr=bool(parsed.get("need_gene_corr", False)),
            gene_list=list(parsed.get("gene_list") or []),
            n_hvg=int(parsed.get("n_hvg", DEFAULT_N_HVG)),
            gene_corr_thr=float(parsed.get("gene_corr_thr", DEFAULT_GENE_CORR_THR)),
            gene_corr_topk=int(parsed.get("gene_corr_topk", DEFAULT_GENE_CORR_TOPK)),
            str_batch=str(parsed.get("str_batch", DEFAULT_STR_BATCH)),
            model_dir_cls=str(DEFAULT_MODEL_DIR),
        ).normalized()
    except (TypeError, ValueError):
        return None


async def resolve_single_cell_params(user_text: str) -> SingleCellAnalysisParams:
    llm_params = await _llm_parse_params(user_text)
    if llm_params is not None:
        return llm_params
    return _heuristic_parse_params(user_text)


def detect_batch_column(h5ad_path: str, preferred: str) -> tuple[str, bool]:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs_columns = list(adata.obs.columns)
        if preferred in obs_columns:
            return preferred, True
        for candidate in ("str_batch", "batch", "batch_id", "sample", "orig.ident", "day"):
            if candidate in obs_columns:
                return candidate, True
        return preferred, False
    finally:
        adata.file.close()


def create_subset_h5ad(input_path: str, output_path: str, max_cells: int) -> str:
    if max_cells <= 0:
        return input_path
    source = ad.read_h5ad(input_path, backed="r")
    try:
        try:
            subset = source[:max_cells].to_memory()
        except Exception:
            source.file.close()
            source = ad.read_h5ad(input_path)
            subset = source[:max_cells].copy()
    finally:
        if getattr(source, "file", None) is not None:
            source.file.close()
    subset.write_h5ad(output_path)
    return output_path


def _add_summary_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.95
    fig.text(0.08, y, title, fontsize=18, fontweight="bold", va="top")
    y -= 0.05
    for line in lines:
        fig.text(0.08, y, line, fontsize=11, va="top")
        y -= 0.035
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            y = 0.95
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    if not image_path.exists():
        return
    image = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.imshow(image)
    ax.set_title(title, fontsize=16, pad=16)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf_report(
    pdf_path: str,
    *,
    user_text: str,
    input_h5ad: str,
    effective_h5ad: str,
    params: SingleCellAnalysisParams,
    result_payload: dict[str, Any],
) -> str:
    pdf_output = Path(pdf_path)
    summary_lines = [
        f"User request: {user_text or '(empty)'}",
        f"Input h5ad: {input_h5ad}",
        f"Executed h5ad: {effective_h5ad}",
        f"Model dir: {params.model_dir_cls}",
        f"need_batch_correction: {params.need_batch_correction}",
        f"need_gene_corr: {params.need_gene_corr}",
        f"gene_list: {', '.join(params.gene_list or []) or '(none)'}",
        f"n_hvg: {params.n_hvg}",
        f"gene_corr_thr: {params.gene_corr_thr}",
        f"gene_corr_topk: {params.gene_corr_topk}",
        f"str_batch: {params.str_batch}",
        f"used_batch_correction: {bool(result_payload.get('used_batch_correction'))}",
        f"runtime_device: {(result_payload.get('runtime') or {}).get('device')}",
        f"result_h5ad: {result_payload.get('result_h5ad') or '(missing)'}",
    ]
    with PdfPages(pdf_output) as pdf:
        _add_summary_page(pdf, "Single-Cell Analysis Report", summary_lines)
        for key, title in (
            ("umap_celltype_png", "UMAP by Predicted Cell Type"),
            ("umap_batch_png", "UMAP by Batch"),
            ("gene_corr_png", "Gene Correlation Network"),
        ):
            path_text = result_payload.get(key)
            if path_text:
                _add_image_page(pdf, Path(path_text), title)
    return str(pdf_output)


def build_report_context_text(
    *,
    user_text: str,
    input_h5ad: str,
    effective_h5ad: str,
    params: SingleCellAnalysisParams,
    result_payload: dict[str, Any],
) -> str:
    runtime = result_payload.get("runtime") or {}
    return build_single_cell_report_context(
        {
            "user_text": user_text or "(empty)",
            "input_h5ad": input_h5ad,
            "effective_h5ad": effective_h5ad,
            "model_dir": params.model_dir_cls,
            "need_batch_correction": params.need_batch_correction,
            "need_gene_corr": params.need_gene_corr,
            "gene_list": ", ".join(params.gene_list or []) or "(none)",
            "n_hvg": params.n_hvg,
            "gene_corr_thr": params.gene_corr_thr,
            "gene_corr_topk": params.gene_corr_topk,
            "str_batch": params.str_batch,
            "used_batch_correction": bool(result_payload.get("used_batch_correction")),
            "runtime_device": runtime.get("device", "unknown"),
            "result_h5ad": result_payload.get("result_h5ad") or "(missing)",
        }
    )


def _build_result_message(
    *,
    params: SingleCellAnalysisParams,
    analysis_payload: dict[str, Any],
    pdf_name: str,
) -> str:
    runtime = analysis_payload.get("runtime") or {}
    lines = [
        "Single-cell analysis completed.",
        f"PDF report: {pdf_name}",
        f"Runtime device: {runtime.get('device', 'unknown')}",
        f"Visible GPU count: {runtime.get('visible_gpu_count', 0)}",
        f"Batch correction requested: {params.need_batch_correction}",
        f"Batch correction used: {bool(analysis_payload.get('used_batch_correction'))}",
        f"Gene correlation requested: {params.need_gene_corr}",
    ]
    if params.need_gene_corr:
        lines.append(f"Genes: {', '.join(params.gene_list or []) or '(none)'}")
    result_h5ad = analysis_payload.get("result_h5ad")
    if result_h5ad:
        lines.append(f"Result h5ad: {result_h5ad}")
    return "\n".join(lines)


async def run_single_cell_skill(
    *,
    user_id: str,
    session_id: str,
    user_text: str,
    h5ad_path: str,
    subset_test_cells: int | None = None,
) -> dict[str, Any]:
    params = (await resolve_single_cell_params(user_text)).normalized()
    detected_batch, has_batch = await asyncio.to_thread(
        detect_batch_column,
        h5ad_path,
        params.str_batch,
    )
    if has_batch:
        params = SingleCellAnalysisParams(
            **{**asdict(params), "str_batch": detected_batch}
        ).normalized()

    session_root = _session_root(user_id, session_id)
    run_root = session_root / "artifacts" / "single_cell_analysis"
    run_root.mkdir(parents=True, exist_ok=True)

    effective_h5ad = h5ad_path
    if subset_test_cells:
        effective_h5ad = await asyncio.to_thread(
            create_subset_h5ad,
            h5ad_path,
            str(run_root / f"subset_{subset_test_cells}.h5ad"),
            subset_test_cells,
        )

    if params.need_gene_corr and len(params.gene_list or []) < 2:
        return {
            "status": "error",
            "message": "Gene correlation requires at least two genes.",
            "analysis_params": asdict(params),
        }

    analysis_payload = await asyncio.to_thread(
        run_analysis_to_dir,
        effective_h5ad,
        str(run_root),
        params.model_dir_cls,
        params.need_batch_correction,
        params.need_gene_corr,
        params.gene_list or [],
        params.n_hvg,
        51,
        params.str_batch,
        128,
        True,
        "spearman",
        params.gene_corr_thr,
        params.gene_corr_topk,
        None,
    )

    pdf_path = await asyncio.to_thread(
        build_pdf_report,
        str(run_root / "single_cell_report.pdf"),
        user_text=user_text,
        input_h5ad=h5ad_path,
        effective_h5ad=effective_h5ad,
        params=params,
        result_payload=analysis_payload,
    )
    report_context = build_report_context_text(
        user_text=user_text,
        input_h5ad=h5ad_path,
        effective_h5ad=effective_h5ad,
        params=params,
        result_payload=analysis_payload,
    )
    report_context_path = run_root / "report_context.txt"
    report_context_path.write_text(report_context, encoding="utf-8")
    relative_pdf = Path(pdf_path).relative_to(session_root)
    pdf_url = (
        f"/api/users/{_safe_user_id(user_id)}/sessions/{session_id}/artifacts/"
        f"{relative_pdf.as_posix()}"
    )

    result_message = _build_result_message(
        params=params,
        analysis_payload=analysis_payload,
        pdf_name=Path(pdf_path).name,
    )
    artifacts: list[dict[str, Any]] = [
        {
            "kind": "pdf",
            "name": Path(pdf_path).name,
            "path": pdf_path,
            "url": pdf_url,
        }
    ]
    return {
        "status": "ok",
        "answer": result_message,
        "local_answer": result_message,
        "analysis_params": asdict(params),
        "analysis_result": analysis_payload,
        "report_context": report_context,
        "artifacts": artifacts,
        "pdf_report": {
            "name": Path(pdf_path).name,
            "path": pdf_path,
            "url": pdf_url,
        },
        "subset_test_cells": subset_test_cells,
    }
