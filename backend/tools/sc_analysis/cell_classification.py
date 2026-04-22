from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.distributed as dist
from scipy.sparse import issparse

from scgpt_source.model import TransformerModel
from scgpt_source.preprocess import preprocess_adata
from scgpt_source.tokenizer import tokenize_and_pad_batch
from scgpt_source.tokenizer.gene_tokenizer import GeneVocab
from scgpt_source.utils import set_seed


def init_distributed_if_needed():
    env_ready = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and ("LOCAL_RANK" in os.environ)
    multi_gpu_ready = torch.cuda.is_available() and torch.cuda.device_count() > 1
    use_ddp = env_ready and multi_gpu_ready
    if not use_ddp:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.cuda.device_count():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return True, rank, world_size, local_rank, device


def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ensure_gene_name(adata: sc.AnnData, gene_col: str = "gene_name") -> sc.AnnData:
    if gene_col not in adata.var.columns:
        adata.var[gene_col] = adata.var_names.astype(str)
    return adata


def load_celltype_map_tsv(tsv_path: str):
    df = pd.read_csv(tsv_path, sep="\t")
    df["celltype_id"] = df["celltype_id"].astype(int)
    df["celltype"] = df["celltype"].astype(str)
    n_cls = int(df["celltype_id"].nunique())
    if set(df["celltype_id"].tolist()) != set(range(n_cls)):
        raise ValueError(f"{tsv_path}: celltype_id 不是 0..{n_cls-1} 连续整数")
    id2name = dict(zip(df["celltype_id"].tolist(), df["celltype"].tolist()))
    return n_cls, id2name


def filter_by_vocab(adata: sc.AnnData, vocab: GeneVocab, gene_col: str = "gene_name") -> sc.AnnData:
    genes = adata.var[gene_col].astype(str).tolist()
    in_vocab = np.array([g in vocab for g in genes], dtype=bool)
    return adata[:, in_vocab].copy()


def get_dense_layer(adata: sc.AnnData, key: str) -> np.ndarray:
    X = adata.layers[key]
    return X.toarray() if issparse(X) else np.asarray(X)


@torch.no_grad()
def encode_in_chunks(
    model,
    gene_ids_cpu,
    values_cpu,
    pad_id,
    chunk_bs,
    amp,
    device,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    outs = []
    N = int(gene_ids_cpu.size(0))
    total_batches = max(1, (N + chunk_bs - 1) // chunk_bs)
    for s in range(0, N, chunk_bs):
        e = min(N, s + chunk_bs)
        batch_idx = (s // chunk_bs) + 1
        g = gene_ids_cpu[s:e].to(device, non_blocking=True)
        v = values_cpu[s:e].to(device, non_blocking=True)
        m = g.eq(pad_id)
        with torch.amp.autocast(device_type="cuda", enabled=amp):
            emb = model.encode_batch(
                src=g,
                values=v.float(),
                src_key_padding_mask=m,
                batch_size=g.size(0),
                batch_labels=None,
                time_step=0,
                return_np=False,
            )
        outs.append(emb.detach().cpu())
        if progress_cb is not None:
            progress_cb(
                {
                    "stage": "celltype_classification",
                    "message": "细胞类型分类推理中",
                    "percent": (100.0 * batch_idx / total_batches),
                    "metrics": {
                        "batch": int(batch_idx),
                        "total_batches": int(total_batches),
                        "processed_cells": int(e),
                        "total_cells": int(N),
                    },
                }
            )
        del g, v, m, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(outs) == 0:
        return torch.empty((0, model.d_model), dtype=torch.float32)
    return torch.cat(outs, dim=0)


def build_cls_model_from_dir(model_dir: Path, vocab: GeneVocab, n_cls: int, pad_token: str, pad_value: int, device):
    def normalize_state_dict(state: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in state.items():
            mapped_key = key
            if ".self_attn.Wqkv.weight" in key:
                mapped_key = key.replace(".self_attn.Wqkv.weight", ".self_attn.in_proj_weight")
            elif ".self_attn.Wqkv.bias" in key:
                mapped_key = key.replace(".self_attn.Wqkv.bias", ".self_attn.in_proj_bias")
            normalized[mapped_key] = value
        return normalized

    args = json.loads((model_dir / "args.json").read_text(encoding="utf-8"))
    use_fast_transformer = device.type == "cuda"
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=args["embsize"],
        nhead=args["nheads"],
        d_hid=args["d_hid"],
        nlayers=args["nlayers"],
        nlayers_cls=args.get("nlayers_cls", 3),
        n_cls=n_cls,
        vocab=vocab,
        dropout=args.get("dropout", 0.2),
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=args.get("do_mvc", False),
        GEP=args.get("GEP", False),
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=0,
        domain_spec_batchnorm=False,
        input_emb_style=args.get("input_emb_style", "continuous"),
        n_input_bins=args.get("n_bins", 51),
        cell_emb_style=args.get("cell_emb_style", "cls"),
        ecs_threshold=args.get("ecs_thres", 0.8),
        explicit_zero_prob=args.get("explicit_zero_prob", False),
        pre_norm=args.get("pre_norm", False),
        gated_attn=args.get("gated_attn", True),
        CLS=True,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
    )
    state = torch.load(model_dir / "best_model.pt", map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    state = normalize_state_dict(state)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        current_state = model.state_dict()
        matched_state = {
            key: value
            for key, value in state.items()
            if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
        }
        current_state.update(matched_state)
        model.load_state_dict(current_state, strict=False)
    model.to(device).eval()
    return model


def run_celltype_classification(
    adata_path: str,
    save_dir: str,
    model_dir_cls: str,
    n_hvg: int,
    n_bins: int = 51,
    use_key: str = "X",
    input_layer_key: str = "X_binned",
    include_zero_gene: bool = True,
    pad_token: str = "<pad>",
    pad_value: int = -2,
    batch_size: int = 128,
    amp: bool = True,
    is_raw_data: bool = True,
    seed: int = 0,
    celltype_map_name: str = "celltype_mapping.tsv",
    output_h5ad: str | None = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    set_seed(seed)
    is_ddp, rank, world_size, _, device = init_distributed_if_needed()
    is_main = (rank == 0)

    save_dir_path = Path(save_dir)
    if is_main:
        save_dir_path.mkdir(parents=True, exist_ok=True)
    ddp_barrier()

    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()
    adata = ensure_gene_name(adata, "gene_name")

    preprocess_dir = save_dir_path / f"rank{rank}" if is_ddp else save_dir_path
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    adata = preprocess_adata(
        adata,
        save_dir=preprocess_dir,
        use_key=use_key,
        ori_batch_col=None,
        n_hvg=n_hvg,
        n_bins=n_bins,
        is_raw_data=is_raw_data,
    )
    adata.layers[input_layer_key] = np.asarray(adata.layers[input_layer_key])

    model_dir = Path(model_dir_cls)
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for tok in [pad_token, "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab[pad_token])

    n_cls, id2name = load_celltype_map_tsv(str(model_dir / celltype_map_name))
    adata = filter_by_vocab(adata, vocab, "gene_name")

    counts = get_dense_layer(adata, input_layer_key)
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
        include_zero_gene=include_zero_gene,
    )
    gene_ids_cpu = tokenized["genes"].cpu()
    values_cpu = tokenized["values"].cpu()
    pad_id = vocab[pad_token]

    model = build_cls_model_from_dir(model_dir, vocab, n_cls, pad_token, pad_value, device)

    n_cells = int(gene_ids_cpu.size(0))
    all_idx = np.arange(n_cells, dtype=int)
    rank_idx = np.array_split(all_idx, world_size)[rank]
    if rank_idx.size > 0:
        if is_main and progress_cb is not None:
            progress_cb(
                {
                    "stage": "celltype_classification",
                    "message": "细胞类型分类推理中",
                    "percent": 0.0,
                    "metrics": {
                        "batch": 0,
                        "total_batches": int(max(1, (rank_idx.size + batch_size - 1) // batch_size)),
                        "processed_cells": 0,
                        "total_cells": int(rank_idx.size),
                    },
                }
            )
        idx_t = torch.from_numpy(rank_idx).long()
        emb_local = encode_in_chunks(
            model,
            gene_ids_cpu.index_select(0, idx_t),
            values_cpu.index_select(0, idx_t),
            pad_id,
            batch_size,
            amp,
            device,
            progress_cb=(progress_cb if is_main else None),
        )
        logits_local = model.cls_decoder(emb_local.to(device)).detach().cpu().numpy()
        pred_local = logits_local.argmax(axis=1).astype(int)
        emb_local_np = emb_local.numpy().astype(np.float32)
    else:
        pred_local = np.zeros((0,), dtype=int)
        emb_local_np = np.zeros((0, model.d_model), dtype=np.float32)

    local_pack = {
        "idx": rank_idx.tolist(),
        "pred": pred_local.tolist(),
        "emb": emb_local_np,
    }

    if is_ddp:
        gathered = [None for _ in range(world_size)] if is_main else None
        dist.gather_object(local_pack, gathered, dst=0)
    else:
        gathered = [local_pack]

    if not is_main:
        ddp_barrier()
        ddp_cleanup()
        return {"adata": None, "cell_emb_cls": None, "n_cls": int(n_cls), "is_main": False}

    emb_dim = next((p["emb"].shape[1] for p in gathered if p["emb"].shape[0] > 0), model.d_model)
    pred_full = np.zeros((n_cells,), dtype=int)
    emb_full = np.zeros((n_cells, emb_dim), dtype=np.float32)
    for p in gathered:
        idx_arr = np.asarray(p["idx"], dtype=int)
        if idx_arr.size == 0:
            continue
        pred_full[idx_arr] = np.asarray(p["pred"], dtype=int)
        emb_full[idx_arr] = np.asarray(p["emb"], dtype=np.float32)

    pred_name = [id2name.get(i, f"Unknown_{i}") for i in pred_full]
    adata.obs["pred_celltype_id"] = pred_full
    adata.obs["pred_celltype"] = np.asarray(pred_name, dtype=object)
    adata.obsm["X_cell_emb_cls"] = emb_full
    if progress_cb is not None:
        progress_cb(
            {
                "stage": "celltype_classification",
                "message": "细胞类型分类推理完成",
                "percent": 100.0,
                "metrics": {"processed_cells": int(n_cells), "total_cells": int(n_cells)},
            }
        )

    if output_h5ad:
        adata.write_h5ad(output_h5ad)

    ddp_barrier()
    ddp_cleanup()
    return {"adata": adata, "cell_emb_cls": emb_full, "n_cls": int(n_cls), "is_main": True}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata_path", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--model_dir_cls", required=True)
    p.add_argument("--n_hvg", type=int, default=1200)
    p.add_argument("--n_bins", type=int, default=51)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--output_h5ad", default="")
    p.add_argument("--payload_json", default="")
    return p.parse_args()


def main():
    args = parse_args()
    out_h5ad = args.output_h5ad or str(Path(args.save_dir) / "classification_result.h5ad")
    res = run_celltype_classification(
        adata_path=args.adata_path,
        save_dir=args.save_dir,
        model_dir_cls=args.model_dir_cls,
        n_hvg=args.n_hvg,
        n_bins=args.n_bins,
        batch_size=args.batch_size,
        output_h5ad=out_h5ad,
    )
    if res.get("is_main"):
        payload = {"classification_h5ad": out_h5ad, "n_cls": int(res["n_cls"])}
        if args.payload_json:
            Path(args.payload_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
