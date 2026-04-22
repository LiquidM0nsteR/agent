from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import issparse

from scgpt_source.model import TransformerModel
from scgpt_source.tokenizer.gene_tokenizer import GeneVocab


def _normalize_scgpt_state_dict_keys(state: dict) -> dict:
    normalized = {}
    for key, value in state.items():
        mapped_key = key
        if ".self_attn.Wqkv.weight" in key:
            mapped_key = key.replace(".self_attn.Wqkv.weight", ".self_attn.in_proj_weight")
        elif ".self_attn.Wqkv.bias" in key:
            mapped_key = key.replace(".self_attn.Wqkv.bias", ".self_attn.in_proj_bias")
        normalized[mapped_key] = value
    return normalized


# -----------------------------
# AnnData / gene / layer helpers
# -----------------------------

def ensure_gene_name(adata: sc.AnnData, gene_col: str = "gene_name") -> sc.AnnData:
    """保证 adata.var[gene_col] 存在；否则用 adata.var_names 填充。"""
    if gene_col not in adata.var.columns:
        adata.var[gene_col] = adata.var_names.astype(str)
    return adata


def load_celltype_map_tsv(tsv_path: str) -> Tuple[int, dict]:
    """
    读取带表头的 tsv：
      celltype_id\tcelltype
      0\tAleurone
      ...

    返回：
      n_cls: int
      id2name: dict[int,str]
    """
    df = pd.read_csv(tsv_path, sep="\t")
    if "celltype_id" not in df.columns or "celltype" not in df.columns:
        raise ValueError(f"{tsv_path} 必须包含列 celltype_id 和 celltype（含表头）")

    df["celltype_id"] = df["celltype_id"].astype(int)
    df["celltype"] = df["celltype"].astype(str)

    n_cls = int(df["celltype_id"].nunique())
    # 强校验：0..n_cls-1 连续
    if set(df["celltype_id"].tolist()) != set(range(n_cls)):
        raise ValueError(f"{tsv_path}: celltype_id 不是 0..{n_cls-1} 连续整数，无法安全映射。")

    id2name = dict(zip(df["celltype_id"].tolist(), df["celltype"].tolist()))
    return n_cls, id2name


def filter_by_vocab(adata: sc.AnnData, vocab: GeneVocab, gene_col: str = "gene_name") -> sc.AnnData:
    """过滤掉不在 vocab 里的基因。"""
    if gene_col not in adata.var.columns:
        adata = ensure_gene_name(adata, gene_col)

    genes = adata.var[gene_col].astype(str).tolist()
    in_vocab = np.array([g in vocab for g in genes], dtype=bool)

    # 可选：你想日志里看到 unmapped 数量，可以打开
    # unmapped = adata.var.loc[~in_vocab, gene_col].tolist()
    # if len(unmapped) > 0:
    #     print(f"[warn] unmapped genes: {len(unmapped)} (show first 20): {unmapped[:20]}")

    return adata[:, in_vocab].copy()


def get_dense_layer(adata: sc.AnnData, key: str) -> np.ndarray:
    """获取 adata.layers[key] 并转 dense numpy。"""
    if key not in adata.layers:
        raise KeyError(f"adata.layers 缺少 {key}")
    X = adata.layers[key]
    return X.A if issparse(X) else np.asarray(X)


# -----------------------------
# Model inference helpers
# -----------------------------

@torch.no_grad()
def encode_in_chunks(
    model: TransformerModel,
    gene_ids_cpu: torch.Tensor,
    values_cpu: torch.Tensor,
    pad_id: int,
    chunk_bs: int,
    amp: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    分块推理，避免一次性把全量 token 搬到 GPU。
    gene_ids_cpu / values_cpu: CPU tensor [N, L]
    返回：CPU tensor [N, D]
    """
    model.eval()
    outs = []
    N = int(gene_ids_cpu.size(0))

    for s in range(0, N, chunk_bs):
        e = min(N, s + chunk_bs)

        g = gene_ids_cpu[s:e].to(device, non_blocking=True)
        v = values_cpu[s:e].to(device, non_blocking=True)
        mask = g.eq(pad_id)

        with torch.amp.autocast(device_type="cuda", enabled=amp):
            emb = model.encode_batch(
                src=g,
                values=v.float(),
                src_key_padding_mask=mask,
                batch_size=g.size(0),
                batch_labels=None,
                time_step=0,
                return_np=False,
            )

        outs.append(emb.detach().cpu())

        del g, v, mask, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(outs, dim=0)


# -----------------------------
# Plot helpers
# -----------------------------

def save_umap_png(
    adata: sc.AnnData,
    outpath: Path,
    color: str,
    title: str = "",
    dpi: int = 300,
) -> None:
    """保存 scanpy UMAP 图。"""
    fig = sc.pl.umap(adata, color=color, show=False, return_fig=True, title=title)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_corr_heatmap(
    genes: List[str],
    corr: np.ndarray,
    outpath: Path,
    dpi: int = 300,
) -> None:
    """保存相关性热图。"""
    n = len(genes)
    # 随着基因数量自适应画布
    w = max(4.0, 0.55 * n)
    h = max(4.0, 0.55 * n)
    plt.figure(figsize=(w, h))
    plt.imshow(corr)
    plt.xticks(range(n), genes, rotation=90, fontsize=8)
    plt.yticks(range(n), genes, fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_gene_corr_network_png(
    genes: List[str],
    corr: np.ndarray,
    outpath: Path,
    threshold: float = 0.3,
    topk: int = 10,
    layout_k: float = 0.6,
    layout_iters: int = 160,
    seed: int = 0,
    dpi: int = 300,
) -> None:
    """保存基因相关性网络图（Top-k 稀疏化 + 强边阈值高亮）。"""
    n = len(genes)
    if n < 2:
        raise ValueError("genes 数量不足，无法构建网络图")

    sim = np.asarray(corr, dtype=np.float32).copy()
    np.fill_diagonal(sim, -np.inf)
    k = max(1, min(int(topk), n - 1))

    edges = []
    for i in range(n):
        idx = np.argpartition(-sim[i], k)[:k]
        for j in idx:
            w = float(sim[i, j])
            if np.isfinite(w) and w > 0:
                edges.append((genes[i], genes[j], w))

    # 去重无向边：保留最大权重
    edge_best = {}
    for a, b, w in edges:
        u, v = (a, b) if a < b else (b, a)
        key = (u, v)
        if key not in edge_best or w > edge_best[key]:
            edge_best[key] = w

    g = nx.Graph()
    for gene in genes:
        g.add_node(gene)
    for (u, v), w in edge_best.items():
        g.add_edge(u, v, weight=w)

    pos = nx.spring_layout(g, k=layout_k, iterations=layout_iters, seed=seed)

    plt.figure(figsize=(14, 14))
    ax = plt.gca()

    elarge = [(u, v) for u, v, d in g.edges(data=True) if d["weight"] >= threshold]
    esmall = [(u, v) for u, v, d in g.edges(data=True) if d["weight"] < threshold]

    nx.draw_networkx_edges(g, pos, edgelist=esmall, width=1.0, edge_color="lightblue", alpha=0.4, ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=elarge, width=3.0, edge_color="blue", alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(g, pos, node_size=200, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(
        g,
        pos,
        font_size=10,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.6),
        ax=ax,
    )

    edge_labels = {(u, v): round(g[u][v]["weight"], 2) for (u, v) in elarge}
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
        ax=ax,
    )

    plt.title(f"Gene Correlation Network (top-k={k}, strong edge >= {threshold})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()


# -----------------------------
# Gene correlation (embedding)
# -----------------------------

def compute_gene_corr(
    adata: sc.AnnData,
    gene_list: List[str],
    model_dir_cls: str,
    layer: Optional[str] = None,
    method: str = "spearman",
) -> Tuple[List[str], np.ndarray]:
    """
    基于模型 encoder(gene_ids) 的基因嵌入相关性。

    参数：
      gene_list: 前端传入的基因名列表（字符串）
      model_dir_cls: 分类模型目录（需包含 args.json / best_model.pt / vocab.json）
      method: "spearman" 或 "pearson"

    返回：
      genes_used: 实际用于计算的基因名（按输入顺序过滤）
      corr: [G, G] float32
    """
    if gene_list is None or len(gene_list) == 0:
        raise ValueError("gene_list 为空")

    gene_list = [str(g) for g in gene_list]

    # 基因名：优先 gene_name，否则 var_names（确保输入数据中存在）
    if "gene_name" in adata.var.columns:
        all_genes = adata.var["gene_name"].astype(str).tolist()
    else:
        all_genes = adata.var_names.astype(str).tolist()
    adata_gene_set = set(all_genes)

    model_dir = Path(model_dir_cls)
    args_path = model_dir / "args.json"
    model_path = model_dir / "best_model.pt"
    vocab_path = model_dir / "vocab.json"
    if not args_path.exists() or not model_path.exists() or not vocab_path.exists():
        raise FileNotFoundError(f"model_dir_cls 缺少必要文件: {model_dir_cls}")

    model_args = json.loads(args_path.read_text(encoding="utf-8"))
    vocab = GeneVocab.from_file(vocab_path)
    for tok in ["<pad>", "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab["<pad>"])

    genes_used = []
    for g in gene_list:
        if (g in adata_gene_set) and (g in vocab):
            genes_used.append(g)

    if len(genes_used) < 2:
        raise ValueError("可用于相关性分析的基因数 < 2（基因名可能不在数据或模型词表中）")

    # 构建模型并加载权重（优先 strict，失败回退 shape-matched）
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
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    state = _normalize_scgpt_state_dict_keys(state)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        cur = model.state_dict()
        filtered = {k: v for k, v in state.items() if (k in cur and v.shape == cur[k].shape)}
        cur.update(filtered)
        model.load_state_dict(cur, strict=False)
    model.eval()

    gene_ids = torch.tensor(vocab(genes_used), dtype=torch.long, device=device)
    with torch.no_grad():
        gene_emb = model.encoder(gene_ids).float().detach().cpu().numpy().astype(np.float32)

    method = method.lower().strip()
    if method not in ("spearman", "pearson"):
        raise ValueError("method 只能是 spearman 或 pearson")

    # [genes, dim]：在 embedding 维度上计算 gene-gene 相关系数
    M = gene_emb
    if method == "spearman":
        # 对每行做 rank transform
        M = M.argsort(axis=1).argsort(axis=1).astype(np.float32)

    M = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-8)
    corr = (M @ M.T) / max(1, (M.shape[1] - 1))
    corr = np.clip(corr, -1.0, 1.0).astype(np.float32)

    return genes_used, corr
