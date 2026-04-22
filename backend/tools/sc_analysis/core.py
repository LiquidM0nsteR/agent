from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scanpy as sc
import torch

from .batch_correction import (
    run_batch_correction_finetune_and_return_emb,
)
from .cell_classification import run_celltype_classification
from .utils import (
    compute_gene_corr,
    save_gene_corr_network_png,
    save_umap_png,
)


ProgressCallback = Callable[[dict[str, Any]], None]


def _visible_gpu_count() -> int:
    return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _run_celltype_classification_auto(
    adata_path: str,
    save_dir: Path,
    model_dir_cls: str,
    n_hvg: int,
    n_bins: int,
    batch_size: int,
    amp: bool,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    n_gpu = _visible_gpu_count()
    can_torchrun = shutil.which("torchrun") is not None

    if n_gpu > 1 and can_torchrun:
        if progress_cb:
            progress_cb(
                {
                    "stage": "celltype_classification",
                    "message": "Running cell-type classification with torchrun.",
                    "metrics": {"mode": "torchrun", "n_gpu": n_gpu},
                }
            )
        save_dir.mkdir(parents=True, exist_ok=True)
        output_h5ad = save_dir / "classification_result.h5ad"
        payload_json = save_dir / "classification_payload.json"
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={n_gpu}",
            "-m",
            "agent.backend.tools.sc_analysis.cell_classification",
            "--adata_path",
            adata_path,
            "--save_dir",
            str(save_dir),
            "--model_dir_cls",
            model_dir_cls,
            "--n_hvg",
            str(n_hvg),
            "--n_bins",
            str(n_bins),
            "--batch_size",
            str(batch_size),
            "--output_h5ad",
            str(output_h5ad),
            "--payload_json",
            str(payload_json),
        ]
        _run_cmd(cmd, _repo_root())
        adata_cls = sc.read_h5ad(output_h5ad)
        cell_emb = np.asarray(adata_cls.obsm["X_cell_emb_cls"], dtype=np.float32)
        return {"adata": adata_cls, "cell_emb_cls": cell_emb}

    return run_celltype_classification(
        adata_path=adata_path,
        save_dir=str(save_dir),
        model_dir_cls=model_dir_cls,
        n_hvg=n_hvg,
        n_bins=n_bins,
        input_layer_key="X_binned",
        batch_size=batch_size,
        amp=amp,
        progress_cb=progress_cb,
    )


def _run_batch_correction_auto(
    adata_path: str,
    save_dir: Path,
    model_dir_cls: str,
    str_batch: str,
    n_hvg: int,
    n_bins: int,
    batch_size: int,
    amp: bool,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    n_gpu = _visible_gpu_count()
    can_torchrun = shutil.which("torchrun") is not None

    if n_gpu > 1 and can_torchrun:
        if progress_cb:
            progress_cb(
                {
                    "stage": "batch_correction_train",
                    "message": "Running batch correction with torchrun.",
                    "metrics": {"mode": "torchrun", "n_gpu": n_gpu},
                }
            )
        save_dir.mkdir(parents=True, exist_ok=True)
        payload_json = save_dir / "bc_payload.json"
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={n_gpu}",
            "-m",
            "agent.backend.tools.sc_analysis.batch_correction",
            "--adata",
            adata_path,
            "--save_dir",
            str(save_dir),
            "--base_model_dir",
            model_dir_cls,
            "--str_batch",
            str_batch,
            "--n_hvg",
            str(n_hvg),
            "--n_bins",
            str(n_bins),
            "--batch_size",
            str(batch_size),
            "--payload_json",
            str(payload_json),
        ]
        if not amp:
            cmd.append("--no_amp")
        _run_cmd(cmd, _repo_root())
        return json.loads(payload_json.read_text(encoding="utf-8"))

    return run_batch_correction_finetune_and_return_emb(
        adata_path=adata_path,
        save_dir=str(save_dir),
        base_model_dir=model_dir_cls,
        str_batch=str_batch,
        n_hvg=n_hvg,
        n_bins=n_bins,
        batch_size=batch_size,
        amp=amp,
        progress_cb=progress_cb,
    )


def run_analysis_to_dir(
    adata_path: str,
    save_dir: str,
    model_dir_cls: str,
    need_batch_correction: bool,
    need_gene_corr: bool,
    gene_list: list[str],
    n_hvg: int,
    n_bins: int = 51,
    str_batch: str = "str_batch",
    batch_size: int = 128,
    amp: bool = True,
    corr_method: str = "spearman",
    corr_threshold: float = 0.3,
    corr_topk: int = 10,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    def emit(
        stage: str,
        message: str,
        percent: float | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if progress_cb is None:
            return
        event: dict[str, Any] = {"stage": stage, "message": message}
        if percent is not None:
            event["percent"] = float(percent)
        if metrics is not None:
            event["metrics"] = metrics
        progress_cb(event)

    def map_percent(local_percent: float, start: float, end: float) -> float:
        bounded = max(0.0, min(100.0, float(local_percent)))
        return start + (end - start) * (bounded / 100.0)

    emit("prepare_input", "Loading input AnnData.", 5.0)
    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()
    emit(
        "prepare_input",
        "Input AnnData loaded.",
        10.0,
        {"n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars)},
    )

    has_batch = (str_batch in adata.obs.columns) and (adata.obs[str_batch].nunique() > 1)
    use_batch_correction = bool(need_batch_correction and has_batch)
    cls_start, cls_end = 10.0, (45.0 if use_batch_correction else 70.0)

    cls_res = _run_celltype_classification_auto(
        adata_path=adata_path,
        save_dir=save_dir_path / "classification",
        model_dir_cls=model_dir_cls,
        n_hvg=n_hvg,
        n_bins=n_bins,
        batch_size=batch_size,
        amp=amp,
        progress_cb=lambda event: emit(
            event.get("stage", "celltype_classification"),
            event.get("message", "Running cell-type classification."),
            map_percent(float(event.get("percent", 0.0)), cls_start, cls_end),
            event.get("metrics"),
        ),
    )
    emit("celltype_classification", "Cell-type classification completed.", cls_end)

    adata_cls = cls_res["adata"]
    adata.obs["pred_celltype_id"] = adata_cls.obs["pred_celltype_id"].values
    adata.obs["pred_celltype"] = adata_cls.obs["pred_celltype"].values
    adata.obsm["X_cell_emb_cls"] = np.asarray(cls_res["cell_emb_cls"], dtype=np.float32)
    adata.obsm["X_cell_emb"] = adata.obsm["X_cell_emb_cls"]

    payload: dict[str, Any] = {
        "used_batch_correction": False,
        "need_gene_corr": bool(need_gene_corr),
        "n_hvg": int(n_hvg),
        "gene_corr_png": None,
        "umap_batch_png": None,
        "runtime": {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "visible_gpu_count": _visible_gpu_count(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
    }

    genes_used: list[str] = []
    if need_gene_corr:
        if not gene_list:
            raise ValueError("need_gene_corr=True requires a non-empty gene_list.")
        genes_used, corr = compute_gene_corr(
            adata=adata,
            gene_list=gene_list,
            model_dir_cls=model_dir_cls,
            layer=None,
            method=corr_method,
        )
        gene_corr_path = save_dir_path / "gene_corr_network.png"
        save_gene_corr_network_png(
            genes=genes_used,
            corr=corr,
            outpath=gene_corr_path,
            threshold=corr_threshold,
            topk=corr_topk,
        )
        payload["gene_corr_png"] = str(gene_corr_path)

    if use_batch_correction:
        bc_start, bc_end = 45.0, 85.0
        bc_payload = _run_batch_correction_auto(
            adata_path=adata_path,
            save_dir=save_dir_path / "batch_correction",
            model_dir_cls=model_dir_cls,
            str_batch=str_batch,
            n_hvg=n_hvg,
            n_bins=n_bins,
            batch_size=batch_size,
            amp=amp,
            progress_cb=lambda event: emit(
                event.get("stage", "batch_correction_train"),
                event.get("message", "Running batch correction."),
                map_percent(float(event.get("percent", 0.0)), bc_start, bc_end),
                event.get("metrics"),
            ),
        )
        emb_path = bc_payload.get("cell_emb_bc_npy")
        if bc_payload.get("trained") and emb_path:
            adata.obsm["X_cell_emb_bc"] = np.load(emb_path).astype(np.float32)
            adata.obsm["X_cell_emb"] = adata.obsm["X_cell_emb_bc"]
            payload["used_batch_correction"] = True
            payload["batch_correction_meta"] = {
                "trained": bool(bc_payload.get("trained")),
                "n_batches": int(bc_payload.get("n_batches", 0)),
                "best_epoch": bc_payload.get("best_epoch"),
                "best_val_loss": bc_payload.get("best_val_loss"),
            }
        emit(
            "batch_correction_train",
            "Batch correction completed.",
            bc_end,
            payload.get("batch_correction_meta"),
        )

    emit("umap", "Computing neighborhood graph and UMAP.", 90.0)
    sc.pp.neighbors(adata, use_rep="X_cell_emb")
    sc.tl.umap(adata)

    umap_celltype_path = save_dir_path / "umap_pred_celltype.png"
    save_umap_png(
        adata=adata,
        outpath=umap_celltype_path,
        color="pred_celltype",
        title="UMAP (predicted cell type)",
    )
    payload["umap_celltype_png"] = str(umap_celltype_path)

    if payload["used_batch_correction"]:
        umap_batch_path = save_dir_path / "umap_batch.png"
        save_umap_png(
            adata=adata,
            outpath=umap_batch_path,
            color=str_batch,
            title="UMAP (batch)",
        )
        payload["umap_batch_png"] = str(umap_batch_path)

    emit("save_result", "Saving analysis outputs.", 97.0)
    result_h5ad_path = save_dir_path / "result.h5ad"
    adata.uns["analysis_meta"] = {
        "need_batch_correction": bool(need_batch_correction),
        "used_batch_correction": bool(payload["used_batch_correction"]),
        "has_batch": bool(has_batch),
        "need_gene_corr": bool(need_gene_corr),
        "genes_used_for_corr": genes_used,
        "n_hvg": int(n_hvg),
        "gene_corr_thr": float(corr_threshold),
        "gene_corr_topk": int(corr_topk),
    }
    adata.write_h5ad(result_h5ad_path)

    payload["result_h5ad"] = str(result_h5ad_path)
    emit("completed", "Analysis completed.", 100.0)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input h5ad path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--model_dir_cls", required=True, help="Cell-type model directory")
    parser.add_argument("--need_batch_correction", default="0", choices=["0", "1"])
    parser.add_argument("--need_gene_corr", default="0", choices=["0", "1"])
    parser.add_argument("--gene_list", default="[]", help="JSON list string")
    parser.add_argument("--n_hvg", type=int, default=1200)
    parser.add_argument("--gene_corr_thr", type=float, default=0.3)
    parser.add_argument("--gene_corr_topk", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        gene_list = json.loads(args.gene_list) if args.gene_list else []
        if not isinstance(gene_list, list):
            gene_list = []
    except Exception:
        gene_list = []

    payload = run_analysis_to_dir(
        adata_path=args.input,
        save_dir=args.outdir,
        model_dir_cls=args.model_dir_cls,
        need_batch_correction=(args.need_batch_correction == "1"),
        need_gene_corr=(args.need_gene_corr == "1"),
        gene_list=[str(gene) for gene in gene_list],
        n_hvg=args.n_hvg,
        corr_threshold=args.gene_corr_thr,
        corr_topk=args.gene_corr_topk,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
