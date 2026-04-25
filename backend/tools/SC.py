from __future__ import annotations

"""
backend/tools/SC.py

单文件版 h5ad 分析工具：
- 输入：h5ad 文件路径，或 AgentInput-like 对象中的 h5ad 附件。
- 输出：PDF 报告路径；在 Agent 流程中返回与 backend/tools/tools.py 一致的 tool_result dict。
- 不再 import backend.tools.sc_analysis 下的 core.py / skill.py / utils.py / cell_classification.py。

说明：
1. 默认执行 scGPT/scGPT-like 细胞类型分类、UMAP 可视化、PDF 汇总报告。
2. 如果模型目录缺少 args.json / best_model.pt / vocab.json / celltype_mapping.tsv，自动降级为 Scanpy 基础 QC + UMAP 报告，避免前端没有返回。
3. 可选 gene correlation 通过 user_text 中的参数开启，例如：
   "做基因相关 gene_list: [OsPIN1, OsPIN2, OsIAA1] gene_corr_thr: 0.3 topk: 10"
4. 批次矫正入口保留 need_batch_correction 参数；本文件默认不做 DAB 微调训练，若需要完整 DAB/DSBN 微调，可再把原 batch_correction.py 的训练循环并入 _run_batch_correction_if_requested。
"""

import argparse
import asyncio
import base64
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse

# -----------------------------------------------------------------------------
# Path bootstrap
# -----------------------------------------------------------------------------

TOOLS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TOOLS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
SCGPT_SOURCE_DIR = BACKEND_DIR / "scgpt_source"

for _path in (BACKEND_DIR, SCGPT_SOURCE_DIR):
    _path_text = str(_path)
    if _path_text not in sys.path:
        sys.path.insert(0, _path_text)

from scgpt_source.model import TransformerModel
from scgpt_source.preprocess import preprocess_adata
from scgpt_source.tokenizer import tokenize_and_pad_batch
from scgpt_source.tokenizer.gene_tokenizer import GeneVocab
from scgpt_source.utils import set_seed

SCGPT_AVAILABLE = True

logger = logging.getLogger(__name__)

TOOL_NAME = "single_cell_analysis"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "scgpt_cls"
DEFAULT_N_HVG = 1200
DEFAULT_N_BINS = 51
DEFAULT_BATCH_SIZE = 128
DEFAULT_STR_BATCH = "str_batch"
DEFAULT_GENE_CORR_THR = 0.3
DEFAULT_GENE_CORR_TOPK = 10

_COMMON_TOOL_RESULT_KEYS = {
    "tool_name",
    "status",
    "query",
    "answer",
    "message",
    "local_answer",
    "evidence_status",
    "references",
    "artifacts",
    "metrics",
    "meta",
    "observation",
}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class SCParams:
    need_batch_correction: bool = False
    need_gene_corr: bool = False
    gene_list: list[str] = field(default_factory=list)
    n_hvg: int = DEFAULT_N_HVG
    n_bins: int = DEFAULT_N_BINS
    str_batch: str = DEFAULT_STR_BATCH
    batch_size: int = DEFAULT_BATCH_SIZE
    gene_corr_thr: float = DEFAULT_GENE_CORR_THR
    gene_corr_topk: int = DEFAULT_GENE_CORR_TOPK
    model_dir_cls: str = str(DEFAULT_MODEL_DIR)
    seed: int = 0
    amp: bool = True

    def normalized(self) -> "SCParams":
        genes: list[str] = []
        seen: set[str] = set()
        for gene in self.gene_list or []:
            gene_text = str(gene).strip()
            if not gene_text:
                continue
            gene_key = gene_text.upper()
            if gene_key in seen:
                continue
            seen.add(gene_key)
            genes.append(gene_text)

        return SCParams(
            need_batch_correction=bool(self.need_batch_correction),
            need_gene_corr=bool(self.need_gene_corr),
            gene_list=genes,
            n_hvg=max(100, int(self.n_hvg or DEFAULT_N_HVG)),
            n_bins=max(2, int(self.n_bins or DEFAULT_N_BINS)),
            str_batch=str(self.str_batch or DEFAULT_STR_BATCH).strip() or DEFAULT_STR_BATCH,
            batch_size=max(1, int(self.batch_size or DEFAULT_BATCH_SIZE)),
            gene_corr_thr=min(1.0, max(0.0, float(self.gene_corr_thr or DEFAULT_GENE_CORR_THR))),
            gene_corr_topk=max(1, int(self.gene_corr_topk or DEFAULT_GENE_CORR_TOPK)),
            model_dir_cls=str(self.model_dir_cls or DEFAULT_MODEL_DIR),
            seed=int(self.seed or 0),
            amp=bool(self.amp),
        )


@dataclass(slots=True)
class SingleCellToolInput:
    user_id: str
    session_id: str
    user_text: str
    h5ad_path: str
    workspace_settings: dict[str, Any] = field(default_factory=dict)

    @property
    def attachments(self) -> list[dict[str, Any]]:
        path = Path(self.h5ad_path).expanduser()
        return [
            {
                "name": path.name,
                "kind": "h5ad",
                "content_type": "application/octet-stream",
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "path": str(path),
            }
        ]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _safe_id(value: str | None, default: str) -> str:
    raw = str(value or "").strip()
    sanitized = "".join(ch for ch in raw if ch.isalnum() or ch in {"-", "_"})
    return sanitized or default


def _session_root(user_id: str, session_id: str) -> Path:
    return PROJECT_ROOT / "data" / "users" / _safe_id(user_id, "anonymous") / "sessions" / _safe_id(session_id, "manual")


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _merge_metrics(*metric_maps: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for metric_map in metric_maps:
        if not isinstance(metric_map, dict):
            continue
        for key, value in metric_map.items():
            if isinstance(value, (int, float)):
                merged[key] = round(float(value), 2)
            elif value is not None:
                merged[key] = value
    return merged


def _build_tool_result(
    *,
    status: str,
    query: str,
    answer: str = "",
    message: str = "",
    evidence_status: str = "not_applicable",
    references: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    metrics: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    **extras: Any,
) -> dict[str, Any]:
    answer_text = str(answer or message or "").strip()
    message_text = str(message or answer_text or "").strip()
    payload: dict[str, Any] = {
        "tool_name": TOOL_NAME,
        "status": str(status or "ok"),
        "query": str(query or ""),
        "answer": answer_text,
        "message": message_text,
        "local_answer": answer_text or message_text,
        "evidence_status": str(evidence_status or "not_applicable"),
        "references": list(references or []),
        "artifacts": list(artifacts or []),
        "metrics": _merge_metrics(metrics),
        "meta": dict(meta or {}),
        "observation": dict(observation or {}),
    }
    for key, value in extras.items():
        if value is not None:
            payload[key] = value
    return payload


def _extract_tool_observation(raw_result: dict[str, Any]) -> dict[str, Any]:
    observation = raw_result.get("observation")
    merged = dict(observation) if isinstance(observation, dict) else {}
    for key, value in raw_result.items():
        if key in _COMMON_TOOL_RESULT_KEYS or value is None:
            continue
        merged[key] = value
    return merged


def _find_h5ad_asset(agent_input: Any) -> Any | None:
    for asset in list(_get_attr(agent_input, "attachments", []) or []):
        kind = str(_get_attr(asset, "kind", "") or "").lower()
        name = str(_get_attr(asset, "name", "") or "")
        path = str(_get_attr(asset, "path", "") or "")
        if kind == "h5ad" or name.lower().endswith(".h5ad") or path.lower().endswith(".h5ad"):
            return asset
    return None


def _resolve_path(path_text: str) -> Path:
    path = Path(str(path_text or "")).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _strip_code_fence(payload: str) -> str:
    text = str(payload or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_bool_from_text(text: str, positive_tokens: tuple[str, ...], negative_tokens: tuple[str, ...]) -> bool | None:
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
        return [chunk.strip() for chunk in re.split(r"[,，\s]+", suffix_match.group(1).strip()) if chunk.strip()]
    return []


def _extract_number(pattern: str, text: str, cast: type[int] | type[float], default: int | float) -> int | float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return default
    try:
        return cast(match.group(1))
    except (TypeError, ValueError):
        return default


def resolve_params_from_text(user_text: str, workspace_settings: dict[str, Any] | None = None) -> SCParams:
    settings = dict(workspace_settings or {})
    text = str(user_text or "")

    need_batch = _parse_bool_from_text(
        text,
        ("batch correction", "batch effect", "批次矫正", "去批次", "批次校正", "消除批次效应"),
        ("no batch correction", "不要批次矫正", "无需批次矫正", "不做批次矫正", "不去批次"),
    )
    need_corr = _parse_bool_from_text(
        text,
        ("gene corr", "gene correlation", "基因相关", "相关网络", "共表达"),
        ("no gene corr", "no gene correlation", "不要基因相关", "不做基因相关"),
    )

    str_batch_match = re.search(
        r"(?:str_batch|batch column|批次列)\s*[:：]?\s*([A-Za-z0-9_.-]+)",
        text,
        flags=re.IGNORECASE,
    )

    model_dir = str(settings.get("single_cell_model_dir") or DEFAULT_MODEL_DIR)
    return SCParams(
        need_batch_correction=bool(need_batch),
        need_gene_corr=bool(need_corr),
        gene_list=_extract_gene_list_from_text(text),
        n_hvg=int(settings.get("single_cell_n_hvg") or _extract_number(r"(?:n_hvg|hvg)\s*[:：]?\s*(\d+)", text, int, DEFAULT_N_HVG)),
        n_bins=int(settings.get("single_cell_n_bins") or DEFAULT_N_BINS),
        str_batch=str_batch_match.group(1) if str_batch_match else str(settings.get("single_cell_batch_key") or DEFAULT_STR_BATCH),
        batch_size=int(settings.get("single_cell_batch_size") or DEFAULT_BATCH_SIZE),
        gene_corr_thr=float(_extract_number(r"(?:gene_corr_thr|corr_threshold|相关阈值)\s*[:：]?\s*([0-9]*\.?[0-9]+)", text, float, DEFAULT_GENE_CORR_THR)),
        gene_corr_topk=int(_extract_number(r"(?:gene_corr_topk|corr_topk|topk)\s*[:：]?\s*(\d+)", text, int, DEFAULT_GENE_CORR_TOPK)),
        model_dir_cls=model_dir,
        seed=int(settings.get("single_cell_seed") or 0),
        amp=bool(settings.get("single_cell_amp", True)),
    ).normalized()


# -----------------------------------------------------------------------------
# AnnData / model helpers merged from sc_analysis/utils.py and cell_classification.py
# -----------------------------------------------------------------------------


def ensure_gene_name(adata: sc.AnnData, gene_col: str = "gene_name") -> sc.AnnData:
    if gene_col not in adata.var.columns:
        adata.var[gene_col] = adata.var_names.astype(str)
    return adata


def get_dense_layer(adata: sc.AnnData, key: str) -> np.ndarray:
    if key not in adata.layers:
        raise KeyError(f"adata.layers missing '{key}'.")
    X = adata.layers[key]
    return X.toarray() if issparse(X) else np.asarray(X)


def load_celltype_map_tsv(tsv_path: str) -> tuple[int, dict[int, str]]:
    df = pd.read_csv(tsv_path, sep="\t")
    df["celltype_id"] = df["celltype_id"].astype(int)
    df["celltype"] = df["celltype"].astype(str)
    n_cls = int(df["celltype_id"].nunique())
    if set(df["celltype_id"].tolist()) != set(range(n_cls)):
        raise ValueError(f"{tsv_path}: celltype_id must be contiguous integers from 0 to {n_cls - 1}.")
    return n_cls, dict(zip(df["celltype_id"].tolist(), df["celltype"].tolist()))


def filter_by_vocab(adata: sc.AnnData, vocab: Any, gene_col: str = "gene_name") -> sc.AnnData:
    genes = adata.var[gene_col].astype(str).tolist()
    in_vocab = np.array([gene in vocab for gene in genes], dtype=bool)
    return adata[:, in_vocab].copy()


def _normalize_scgpt_state_dict_keys(state: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        mapped_key = key
        if ".self_attn.Wqkv.weight" in key:
            mapped_key = key.replace(".self_attn.Wqkv.weight", ".self_attn.in_proj_weight")
        elif ".self_attn.Wqkv.bias" in key:
            mapped_key = key.replace(".self_attn.Wqkv.bias", ".self_attn.in_proj_bias")
        normalized[mapped_key] = value
    return normalized


def _required_model_files_exist(model_dir: Path) -> bool:
    required = ["args.json", "best_model.pt", "vocab.json", "celltype_mapping.tsv"]
    return all((model_dir / item).exists() for item in required)


def _load_vocab(model_dir: Path, pad_token: str = "<pad>") -> Any:
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for token in [pad_token, "<cls>", "<eoc>"]:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab[pad_token])
    return vocab


def build_cls_model_from_dir(model_dir: Path, vocab: Any, n_cls: int, pad_token: str, pad_value: int, device: torch.device) -> Any:
    model_args = json.loads((model_dir / "args.json").read_text(encoding="utf-8"))
    use_fast_transformer = device.type == "cuda"

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_args["embsize"],
        nhead=model_args["nheads"],
        d_hid=model_args["d_hid"],
        nlayers=model_args["nlayers"],
        nlayers_cls=model_args.get("nlayers_cls", 3),
        n_cls=n_cls,
        vocab=vocab,
        dropout=model_args.get("dropout", 0.2),
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=model_args.get("do_mvc", False),
        GEP=model_args.get("GEP", False),
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=0,
        domain_spec_batchnorm=False,
        input_emb_style=model_args.get("input_emb_style", "continuous"),
        n_input_bins=model_args.get("n_bins", 51),
        cell_emb_style=model_args.get("cell_emb_style", "cls"),
        ecs_threshold=model_args.get("ecs_thres", 0.8),
        explicit_zero_prob=model_args.get("explicit_zero_prob", False),
        pre_norm=model_args.get("pre_norm", False),
        gated_attn=model_args.get("gated_attn", True),
        CLS=True,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
    )

    state = torch.load(model_dir / "best_model.pt", map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    state = _normalize_scgpt_state_dict_keys(state)

    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        current = model.state_dict()
        matched = {
            key: value
            for key, value in state.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
        current.update(matched)
        model.load_state_dict(current, strict=False)

    model.to(device).eval()
    return model


@torch.no_grad()
def encode_in_chunks(
    model: Any,
    gene_ids_cpu: torch.Tensor,
    values_cpu: torch.Tensor,
    pad_id: int,
    chunk_bs: int,
    amp: bool,
    device: torch.device,
) -> torch.Tensor:
    outs: list[torch.Tensor] = []
    n_cells = int(gene_ids_cpu.size(0))
    autocast_enabled = bool(amp and device.type == "cuda")

    for start in range(0, n_cells, chunk_bs):
        end = min(n_cells, start + chunk_bs)
        genes = gene_ids_cpu[start:end].to(device, non_blocking=True)
        values = values_cpu[start:end].to(device, non_blocking=True)
        padding_mask = genes.eq(pad_id)

        with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
            emb = model.encode_batch(
                src=genes,
                values=values.float(),
                src_key_padding_mask=padding_mask,
                batch_size=genes.size(0),
                batch_labels=None,
                time_step=0,
                return_np=False,
            )

        outs.append(emb.detach().cpu())
        del genes, values, padding_mask, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not outs:
        return torch.empty((0, int(getattr(model, "d_model", 0))), dtype=torch.float32)
    return torch.cat(outs, dim=0)


def run_celltype_classification_in_memory(
    *,
    adata_path: str,
    save_dir: Path,
    model_dir_cls: str,
    n_hvg: int,
    n_bins: int,
    batch_size: int,
    amp: bool,
    seed: int,
) -> dict[str, Any]:
    if not SCGPT_AVAILABLE:
        raise RuntimeError("scgpt_source is not available; cannot run model-based cell-type classification.")

    set_seed(seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(model_dir_cls).expanduser()
    if not _required_model_files_exist(model_dir):
        raise FileNotFoundError(f"model_dir_cls lacks required files: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_token = "<pad>"
    pad_value = -2

    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()
    adata = ensure_gene_name(adata, "gene_name")

    adata = preprocess_adata(
        adata,
        save_dir=save_dir,
        use_key="X",
        ori_batch_col=None,
        n_hvg=n_hvg,
        n_bins=n_bins,
        is_raw_data=True,
    )
    adata.layers["X_binned"] = np.asarray(adata.layers["X_binned"])

    vocab = _load_vocab(model_dir, pad_token=pad_token)
    n_cls, id2name = load_celltype_map_tsv(str(model_dir / "celltype_mapping.tsv"))
    adata = filter_by_vocab(adata, vocab, "gene_name")

    counts = get_dense_layer(adata, "X_binned")
    genes = adata.var["gene_name"].astype(str).tolist()
    gene_ids_arr = np.array(vocab(genes), dtype=int)

    tokenized = tokenize_and_pad_batch(
        counts,
        gene_ids_arr,
        max_len=n_hvg + 1,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    gene_ids_cpu = tokenized["genes"].cpu()
    values_cpu = tokenized["values"].cpu()
    pad_id = vocab[pad_token]

    model = build_cls_model_from_dir(model_dir, vocab, n_cls, pad_token, pad_value, device)
    emb_cpu = encode_in_chunks(model, gene_ids_cpu, values_cpu, pad_id, batch_size, amp, device)

    logits = model.cls_decoder(emb_cpu.to(device)).detach().cpu().numpy()
    pred_ids = logits.argmax(axis=1).astype(int)
    pred_names = [id2name.get(int(item), f"Unknown_{int(item)}") for item in pred_ids]
    cell_emb = emb_cpu.numpy().astype(np.float32)

    adata.obs["pred_celltype_id"] = pred_ids
    adata.obs["pred_celltype"] = np.asarray(pred_names, dtype=object)
    adata.obsm["X_cell_emb_cls"] = cell_emb

    return {
        "adata_cls": adata,
        "cell_emb_cls": cell_emb,
        "n_cls": int(n_cls),
        "device": str(device),
        "model_dir": str(model_dir),
    }


# -----------------------------------------------------------------------------
# Scanpy fallback and plotting
# -----------------------------------------------------------------------------


def run_scanpy_fallback(adata: sc.AnnData, *, n_hvg: int, seed: int) -> sc.AnnData:
    adata = adata.copy()
    adata.var_names_make_unique()
    ensure_gene_name(adata)

    if issparse(adata.X):
        total = np.asarray(adata.X.sum(axis=1)).reshape(-1)
        detected = np.asarray((adata.X > 0).sum(axis=1)).reshape(-1)
    else:
        x = np.asarray(adata.X)
        total = x.sum(axis=1)
        detected = (x > 0).sum(axis=1)

    adata.obs["total_counts"] = total
    adata.obs["n_genes_by_counts"] = detected

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=min(n_hvg, adata.n_vars), flavor="seurat")
        if "highly_variable" in adata.var.columns and int(adata.var["highly_variable"].sum()) > 10:
            adata = adata[:, adata.var["highly_variable"]].copy()
    except Exception:
        pass
    sc.pp.pca(adata, n_comps=min(50, adata.n_obs - 1, adata.n_vars - 1), random_state=seed)
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.umap(adata, random_state=seed)
    adata.obs["pred_celltype"] = "Unknown"
    adata.obs["pred_celltype_id"] = 0
    adata.obsm["X_cell_emb"] = adata.obsm["X_pca"].astype(np.float32)
    return adata


def save_umap_png(adata: sc.AnnData, outpath: Path, color: str, title: str, dpi: int = 300) -> None:
    if "X_umap" not in adata.obsm:
        raise KeyError("adata.obsm['X_umap'] is missing.")
    fig = sc.pl.umap(adata, color=color, show=False, return_fig=True, title=title)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_qc_png(adata: sc.AnnData, outpath: Path, dpi: int = 300) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if "total_counts" in adata.obs.columns:
        axes[0].hist(np.asarray(adata.obs["total_counts"], dtype=float), bins=50)
        axes[0].set_title("Total counts")
    else:
        axes[0].axis("off")
    if "n_genes_by_counts" in adata.obs.columns:
        axes[1].hist(np.asarray(adata.obs["n_genes_by_counts"], dtype=float), bins=50)
        axes[1].set_title("Detected genes")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Gene correlation merged from sc_analysis/utils.py
# -----------------------------------------------------------------------------


def compute_gene_corr(
    *,
    adata: sc.AnnData,
    gene_list: list[str],
    model_dir_cls: str,
    method: str = "spearman",
) -> tuple[list[str], np.ndarray]:
    if not SCGPT_AVAILABLE:
        raise RuntimeError("scgpt_source is not available; cannot compute model gene embeddings.")
    if not gene_list:
        raise ValueError("gene_list is empty.")

    model_dir = Path(model_dir_cls).expanduser()
    if not (model_dir / "args.json").exists() or not (model_dir / "best_model.pt").exists() or not (model_dir / "vocab.json").exists():
        raise FileNotFoundError(f"model_dir_cls lacks required files: {model_dir}")

    if "gene_name" in adata.var.columns:
        adata_genes = set(adata.var["gene_name"].astype(str).tolist())
    else:
        adata_genes = set(adata.var_names.astype(str).tolist())

    model_args = json.loads((model_dir / "args.json").read_text(encoding="utf-8"))
    vocab = _load_vocab(model_dir)
    genes_used = [str(gene) for gene in gene_list if str(gene) in adata_genes and str(gene) in vocab]
    if len(genes_used) < 2:
        raise ValueError("Fewer than two requested genes are available in both AnnData and model vocabulary.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fast_transformer = device.type == "cuda"
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_args["embsize"],
        nhead=model_args["nheads"],
        d_hid=model_args["d_hid"],
        nlayers=model_args["nlayers"],
        nlayers_cls=model_args.get("nlayers_cls", 3),
        n_cls=1,
        vocab=vocab,
        dropout=model_args.get("dropout", 0.2),
        pad_token="<pad>",
        pad_value=model_args.get("pad_value", -2),
        do_mvc=model_args.get("do_mvc", False),
        GEP=model_args.get("GEP", False),
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=0,
        domain_spec_batchnorm=False,
        input_emb_style=model_args.get("input_emb_style", "continuous"),
        n_input_bins=model_args.get("n_bins", 51),
        cell_emb_style=model_args.get("cell_emb_style", "cls"),
        ecs_threshold=model_args.get("ecs_thres", 0.8),
        explicit_zero_prob=model_args.get("explicit_zero_prob", False),
        pre_norm=model_args.get("pre_norm", False),
        gated_attn=model_args.get("gated_attn", True),
        CLS=True,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
    ).to(device)

    state = torch.load(model_dir / "best_model.pt", map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    state = _normalize_scgpt_state_dict_keys(state)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        current = model.state_dict()
        matched = {k: v for k, v in state.items() if k in current and tuple(current[k].shape) == tuple(v.shape)}
        current.update(matched)
        model.load_state_dict(current, strict=False)
    model.eval()

    gene_ids = torch.tensor(vocab(genes_used), dtype=torch.long, device=device)
    with torch.no_grad():
        gene_emb = model.encoder(gene_ids).float().detach().cpu().numpy().astype(np.float32)

    method = method.lower().strip()
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be spearman or pearson.")

    matrix = gene_emb
    if method == "spearman":
        matrix = matrix.argsort(axis=1).argsort(axis=1).astype(np.float32)
    matrix = (matrix - matrix.mean(axis=1, keepdims=True)) / (matrix.std(axis=1, keepdims=True) + 1e-8)
    corr = (matrix @ matrix.T) / max(1, matrix.shape[1] - 1)
    corr = np.clip(corr, -1.0, 1.0).astype(np.float32)
    return genes_used, corr


def save_gene_corr_network_png(
    *,
    genes: list[str],
    corr: np.ndarray,
    outpath: Path,
    threshold: float,
    topk: int,
    seed: int,
    dpi: int = 300,
) -> None:
    n = len(genes)
    if n < 2:
        raise ValueError("At least two genes are required for correlation network plotting.")

    sim = np.asarray(corr, dtype=np.float32).copy()
    np.fill_diagonal(sim, -np.inf)
    k = max(1, min(int(topk), n - 1))

    edge_best: dict[tuple[str, str], float] = {}
    for i in range(n):
        idx = np.argpartition(-sim[i], k)[:k]
        for j in idx:
            weight = float(sim[i, j])
            if not np.isfinite(weight) or weight <= 0:
                continue
            u, v = (genes[i], genes[j]) if genes[i] < genes[j] else (genes[j], genes[i])
            if (u, v) not in edge_best or weight > edge_best[(u, v)]:
                edge_best[(u, v)] = weight

    graph = nx.Graph()
    graph.add_nodes_from(genes)
    for (u, v), weight in edge_best.items():
        graph.add_edge(u, v, weight=weight)

    pos = nx.spring_layout(graph, k=0.6, iterations=160, seed=seed)
    plt.figure(figsize=(14, 14))
    ax = plt.gca()

    strong_edges = [(u, v) for u, v, data in graph.edges(data=True) if data["weight"] >= threshold]
    weak_edges = [(u, v) for u, v, data in graph.edges(data=True) if data["weight"] < threshold]

    nx.draw_networkx_edges(graph, pos, edgelist=weak_edges, width=1.0, edge_color="lightblue", alpha=0.4, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=strong_edges, width=3.0, edge_color="blue", alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_size=220, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=9, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.65), ax=ax)
    edge_labels = {(u, v): round(graph[u][v]["weight"], 2) for (u, v) in strong_edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, rotate=False, ax=ax)
    plt.title(f"Gene Correlation Network (top-k={k}, strong edge >= {threshold})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# PDF report
# -----------------------------------------------------------------------------


def _add_summary_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.95
    fig.text(0.08, y, title, fontsize=18, fontweight="bold", va="top")
    y -= 0.055
    for line in lines:
        fig.text(0.08, y, str(line), fontsize=10.5, va="top")
        y -= 0.032
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
    *,
    pdf_path: Path,
    user_text: str,
    input_h5ad: Path,
    params: SCParams,
    result_payload: dict[str, Any],
) -> str:
    runtime = result_payload.get("runtime") or {}
    summary_lines = [
        f"User request: {user_text or '(empty)'}",
        f"Input h5ad: {input_h5ad}",
        f"n_cells: {result_payload.get('n_cells')}",
        f"n_genes: {result_payload.get('n_genes')}",
        f"mode: {result_payload.get('mode')}",
        f"model_dir: {params.model_dir_cls}",
        f"n_hvg: {params.n_hvg}",
        f"n_bins: {params.n_bins}",
        f"runtime_device: {runtime.get('device', 'unknown')}",
        f"cuda_available: {runtime.get('torch_cuda_available')}",
        f"visible_gpu_count: {runtime.get('visible_gpu_count')}",
        f"need_batch_correction: {params.need_batch_correction}",
        f"used_batch_correction: {result_payload.get('used_batch_correction')}",
        f"need_gene_corr: {params.need_gene_corr}",
        f"gene_list: {', '.join(params.gene_list) if params.gene_list else '(none)'}",
        f"result_h5ad: {result_payload.get('result_h5ad') or '(missing)'}",
    ]
    if result_payload.get("warnings"):
        summary_lines.append("Warnings:")
        summary_lines.extend([f"- {item}" for item in result_payload.get("warnings", [])])

    with PdfPages(pdf_path) as pdf:
        _add_summary_page(pdf, "Single-Cell Analysis Report", summary_lines)
        image_items = [
            ("qc_png", "Quality Control Summary"),
            ("umap_celltype_png", "UMAP by Predicted Cell Type"),
            ("umap_batch_png", "UMAP by Batch"),
            ("gene_corr_png", "Gene Correlation Network"),
        ]
        for key, title in image_items:
            path_text = result_payload.get(key)
            if path_text:
                _add_image_page(pdf, Path(path_text), title)
    return str(pdf_path)


def build_report_context_text(*, user_text: str, input_h5ad: Path, params: SCParams, result_payload: dict[str, Any]) -> str:
    runtime = result_payload.get("runtime") or {}
    lines = [
        "Single-cell analysis finished.",
        f"User request: {user_text or '(empty)'}",
        f"Input h5ad: {input_h5ad}",
        f"Cells: {result_payload.get('n_cells')}",
        f"Genes: {result_payload.get('n_genes')}",
        f"Mode: {result_payload.get('mode')}",
        f"Model dir: {params.model_dir_cls}",
        f"Runtime device: {runtime.get('device', 'unknown')}",
        f"Batch correction requested: {params.need_batch_correction}",
        f"Batch correction used: {result_payload.get('used_batch_correction')}",
        f"Gene correlation requested: {params.need_gene_corr}",
        f"Result h5ad: {result_payload.get('result_h5ad')}",
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Core h5ad -> PDF pipeline
# -----------------------------------------------------------------------------


def detect_batch_column(adata: sc.AnnData, preferred: str) -> tuple[str, bool]:
    obs_columns = list(adata.obs.columns)
    if preferred in obs_columns and adata.obs[preferred].nunique() > 1:
        return preferred, True
    for candidate in ("str_batch", "batch", "batch_id", "sample", "orig.ident", "day"):
        if candidate in obs_columns and adata.obs[candidate].nunique() > 1:
            return candidate, True
    return preferred, False


def analyze_h5ad_to_pdf(
    *,
    h5ad_path: str | Path,
    output_dir: str | Path,
    user_text: str = "",
    params: SCParams | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    input_h5ad = Path(h5ad_path).expanduser().resolve()
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if params is None:
        params = resolve_params_from_text(user_text)
    params = params.normalized()

    if not input_h5ad.exists():
        raise FileNotFoundError(f"h5ad file not found: {input_h5ad}")

    warnings_list: list[str] = []
    original = sc.read_h5ad(input_h5ad)
    original.var_names_make_unique()
    ensure_gene_name(original)
    batch_key, has_batch = detect_batch_column(original, params.str_batch)
    if has_batch and batch_key != params.str_batch:
        params = SCParams(**{**asdict(params), "str_batch": batch_key}).normalized()

    runtime = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "visible_gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    qc_png = outdir / "qc_summary.png"
    try:
        save_qc_png(original, qc_png)
    except Exception as exc:
        warnings_list.append(f"QC plot skipped: {exc}")

    mode = "model_classification"
    used_model = False
    try:
        cls_result = run_celltype_classification_in_memory(
            adata_path=str(input_h5ad),
            save_dir=outdir / "classification",
            model_dir_cls=params.model_dir_cls,
            n_hvg=params.n_hvg,
            n_bins=params.n_bins,
            batch_size=params.batch_size,
            amp=params.amp,
            seed=params.seed,
        )
        adata_cls = cls_result["adata_cls"]
        original.obs["pred_celltype_id"] = adata_cls.obs["pred_celltype_id"].values
        original.obs["pred_celltype"] = adata_cls.obs["pred_celltype"].values
        original.obsm["X_cell_emb_cls"] = np.asarray(cls_result["cell_emb_cls"], dtype=np.float32)
        original.obsm["X_cell_emb"] = original.obsm["X_cell_emb_cls"]
        used_model = True
    except Exception as exc:
        mode = "scanpy_fallback"
        warnings_list.append(f"Model classification skipped; fallback to Scanpy workflow. Reason: {exc}")
        original = run_scanpy_fallback(original, n_hvg=params.n_hvg, seed=params.seed)

    # 批次矫正：这里保留结果字段。完整 DAB 微调训练不在本文件默认启用。
    used_batch_correction = False
    if params.need_batch_correction:
        if not has_batch:
            warnings_list.append(f"Batch correction requested but no multi-batch column found for '{params.str_batch}'.")
        else:
            warnings_list.append(
                "Batch correction requested, but DAB/DSBN finetuning is not enabled in this compact SC.py. "
                "Current report uses classification/fallback embedding."
            )

    sc.pp.neighbors(original, use_rep="X_cell_emb")
    sc.tl.umap(original, random_state=params.seed)

    umap_celltype_png = outdir / "umap_pred_celltype.png"
    save_umap_png(original, umap_celltype_png, color="pred_celltype", title="UMAP (predicted cell type)")

    umap_batch_png: Path | None = None
    if has_batch:
        umap_batch_png = outdir / "umap_batch.png"
        save_umap_png(original, umap_batch_png, color=params.str_batch, title=f"UMAP (batch: {params.str_batch})")

    gene_corr_png: Path | None = None
    genes_used: list[str] = []
    if params.need_gene_corr:
        try:
            genes_used, corr = compute_gene_corr(
                adata=original,
                gene_list=params.gene_list,
                model_dir_cls=params.model_dir_cls,
                method="spearman",
            )
            gene_corr_png = outdir / "gene_corr_network.png"
            save_gene_corr_network_png(
                genes=genes_used,
                corr=corr,
                outpath=gene_corr_png,
                threshold=params.gene_corr_thr,
                topk=params.gene_corr_topk,
                seed=params.seed,
            )
        except Exception as exc:
            warnings_list.append(f"Gene correlation skipped: {exc}")

    result_h5ad = outdir / "result.h5ad"
    original.uns["analysis_meta"] = {
        "mode": mode,
        "used_model": used_model,
        "need_batch_correction": params.need_batch_correction,
        "used_batch_correction": used_batch_correction,
        "has_batch": has_batch,
        "batch_key": params.str_batch,
        "need_gene_corr": params.need_gene_corr,
        "genes_used_for_corr": genes_used,
        "n_hvg": params.n_hvg,
        "gene_corr_thr": params.gene_corr_thr,
        "gene_corr_topk": params.gene_corr_topk,
        "warnings": warnings_list,
    }
    original.write_h5ad(result_h5ad)

    payload: dict[str, Any] = {
        "mode": mode,
        "used_model": used_model,
        "used_batch_correction": used_batch_correction,
        "has_batch": has_batch,
        "batch_key": params.str_batch,
        "need_gene_corr": params.need_gene_corr,
        "genes_used_for_corr": genes_used,
        "n_cells": int(original.n_obs),
        "n_genes": int(original.n_vars),
        "n_hvg": int(params.n_hvg),
        "runtime": runtime,
        "warnings": warnings_list,
        "qc_png": str(qc_png) if qc_png.exists() else None,
        "umap_celltype_png": str(umap_celltype_png),
        "umap_batch_png": str(umap_batch_png) if umap_batch_png else None,
        "gene_corr_png": str(gene_corr_png) if gene_corr_png else None,
        "result_h5ad": str(result_h5ad),
        "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
    }

    pdf_path = outdir / "single_cell_report.pdf"
    build_pdf_report(
        pdf_path=pdf_path,
        user_text=user_text,
        input_h5ad=input_h5ad,
        params=params,
        result_payload=payload,
    )
    report_context = build_report_context_text(
        user_text=user_text,
        input_h5ad=input_h5ad,
        params=params,
        result_payload=payload,
    )
    (outdir / "report_context.txt").write_text(report_context, encoding="utf-8")

    payload["pdf_report"] = str(pdf_path)
    payload["report_context"] = report_context
    return payload


# -----------------------------------------------------------------------------
# Agent-compatible public API
# -----------------------------------------------------------------------------


async def run_sc_analysis_query(agent_input: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    user_id = str(_get_attr(agent_input, "user_id", "anonymous") or "anonymous")
    session_id = str(_get_attr(agent_input, "session_id", "manual") or "manual")
    user_text = str(_get_attr(agent_input, "user_text", "") or "")
    workspace_settings = dict(_get_attr(agent_input, "workspace_settings", {}) or {})

    asset = _find_h5ad_asset(agent_input)
    if asset is None:
        return _build_tool_result(
            status="error",
            query=user_text,
            message="Single-cell analysis requires an h5ad attachment.",
            evidence_status="not_applicable",
            metrics={"tool_ms": round((time.perf_counter() - started_at) * 1000, 2)},
        )

    h5ad_path = _resolve_path(str(_get_attr(asset, "path", "") or ""))
    if not h5ad_path.exists() or not h5ad_path.is_file():
        return _build_tool_result(
            status="error",
            query=user_text,
            message=f"h5ad file not found: {h5ad_path}",
            evidence_status="not_applicable",
            metrics={"tool_ms": round((time.perf_counter() - started_at) * 1000, 2)},
            meta={"input_h5ad": str(h5ad_path)},
        )

    params = resolve_params_from_text(user_text, workspace_settings).normalized()
    run_root = _session_root(user_id, session_id) / "artifacts" / "single_cell_analysis"
    run_root.mkdir(parents=True, exist_ok=True)

    try:
        raw_result = await asyncio.to_thread(
            analyze_h5ad_to_pdf,
            h5ad_path=h5ad_path,
            output_dir=run_root,
            user_text=user_text,
            params=params,
        )
    except Exception as exc:
        logger.exception("single-cell analysis failed")
        return _build_tool_result(
            status="error",
            query=user_text,
            message=f"Single-cell analysis failed: {exc}",
            evidence_status="not_applicable",
            metrics={"tool_ms": round((time.perf_counter() - started_at) * 1000, 2)},
            meta={"input_h5ad": str(h5ad_path)},
        )

    pdf_path = Path(raw_result["pdf_report"])
    session_root = _session_root(user_id, session_id)
    relative_pdf = pdf_path.relative_to(session_root)
    pdf_url = f"/api/users/{_safe_id(user_id, 'anonymous')}/sessions/{_safe_id(session_id, 'manual')}/artifacts/{relative_pdf.as_posix()}"

    answer = "Single-cell analysis completed.\n" f"PDF report: {pdf_path.name}\n" f"Mode: {raw_result.get('mode')}\n" f"Cells: {raw_result.get('n_cells')}, Genes: {raw_result.get('n_genes')}"
    if raw_result.get("warnings"):
        answer += "\nWarnings:\n" + "\n".join(f"- {item}" for item in raw_result.get("warnings", []))

    artifacts = [
        {
            "kind": "pdf",
            "name": pdf_path.name,
            "path": str(pdf_path),
            "url": pdf_url,
        }
    ]

    return _build_tool_result(
        status="ok",
        query=user_text,
        answer=answer,
        message=answer,
        evidence_status="sufficient",
        artifacts=artifacts,
        metrics={"tool_ms": round((time.perf_counter() - started_at) * 1000, 2), "analysis_ms": raw_result.get("elapsed_ms")},
        meta={"input_h5ad": str(h5ad_path), "runner": "backend.tools.SC.run_sc_analysis_query"},
        observation=_extract_tool_observation(raw_result),
        analysis_params=asdict(params),
        analysis_result=raw_result,
        report_context=raw_result.get("report_context"),
        pdf_report={"name": pdf_path.name, "path": str(pdf_path), "url": pdf_url},
    )


run_single_cell_analysis = run_sc_analysis_query
run_sc_analysis = run_sc_analysis_query


# -----------------------------------------------------------------------------
# Direct CLI: python -m backend.tools.SC --h5ad xxx.h5ad
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run h5ad analysis and return a PDF report.")
    parser.add_argument("--h5ad", required=True, help="Input h5ad path")
    parser.add_argument("--outdir", default="", help="Output directory; default: data/manual_sc_runs/<timestamp>")
    parser.add_argument("--text", default="", help="Optional user request text")
    parser.add_argument("--model_dir_cls", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--n_hvg", type=int, default=DEFAULT_N_HVG)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--need_gene_corr", action="store_true")
    parser.add_argument("--gene_list", default="[]", help="JSON list, e.g. '[\"OsPIN1\", \"OsPIN2\"]'")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        gene_list = json.loads(args.gene_list)
        if not isinstance(gene_list, list):
            gene_list = []
    except Exception:
        gene_list = []

    outdir = Path(args.outdir) if args.outdir else PROJECT_ROOT / "data" / "manual_sc_runs" / time.strftime("%Y%m%d-%H%M%S")
    params = SCParams(
        need_gene_corr=bool(args.need_gene_corr),
        gene_list=[str(item) for item in gene_list],
        n_hvg=args.n_hvg,
        batch_size=args.batch_size,
        model_dir_cls=args.model_dir_cls,
    ).normalized()
    result = analyze_h5ad_to_pdf(
        h5ad_path=args.h5ad,
        output_dir=outdir,
        user_text=args.text,
        params=params,
    )
    print(json.dumps({"status": "ok", "pdf_report": result.get("pdf_report"), "result": result}, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
