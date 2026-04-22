#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scweb/backend/batch_correction.py

目标（按你的最新要求）：
- 仅当输入 adata 为多批次（obs[str_batch] 的唯一值 > 1）时：
  1) 执行“批次整合微调”（DAB + DSBN + use_batch_labels + GEP/GEPC/ECS）
  2) 用“微调后的最佳模型”在全体细胞上推理得到 batch-corrected cell embedding：cell_emb_bc
  3) 只返回/保存 cell_emb_bc（不输出/不依赖训练好的模型目录给后续再加载）

输出：
- save_dir / "cell_emb_bc.npy"      (float32, shape [n_cells, d_model])
- save_dir / "result.h5ad"          (写入 adata.obsm["X_cell_emb_bc"])
- stdout 打印 JSON payload，方便后端直接返回：
  {
    "trained": true/false,
    "n_batches": ...,
    "cell_emb_bc_npy": ".../cell_emb_bc.npy" (多批次才有),
    "result_h5ad": ".../result.h5ad"         (多批次才有),
    "best_epoch": ...,
    "best_val_loss": ...
  }

运行（单卡）：
  python scweb/backend/batch_correction.py \
    --adata scweb/data/uploads/adata.h5ad \
    --save_dir scweb/data/results/bc_job \
    --pretrain_dir scgpt/pretrain/pretrain_model_my \
    --str_batch str_batch --n_hvg 1200 --epochs 10

运行（多卡 DDP）：
  torchrun --nproc_per_node=2 scweb/backend/batch_correction.py \
    --adata ... --save_dir ... --pretrain_dir ... --str_batch str_batch
"""

import os
import sys
import json
import time
import gc
import warnings
import argparse
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import numpy as np
import scanpy as sc
from scipy.sparse import issparse

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split

# ---- project imports ----
import scgpt_source as scg
from scgpt_source.model import TransformerModel
from scgpt_source.tokenizer.gene_tokenizer import GeneVocab
from scgpt_source.tokenizer import tokenize_and_pad_batch
from scgpt_source.loss import masked_mse_loss
from scgpt_source import trainer
from scgpt_source.utils import set_seed
from scgpt_source.preprocess import preprocess_adata

warnings.simplefilter("ignore", category=FutureWarning)


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


# -------------------- DDP helpers --------------------

def init_distributed_if_needed():
    env_ready = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and ("LOCAL_RANK" in os.environ)
    multi_gpu_ready = torch.cuda.is_available() and torch.cuda.device_count() > 1
    use_ddp = env_ready and multi_gpu_ready
    if not use_ddp:
        return False, 0, 1, 0
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.cuda.device_count():
        return False, 0, 1, 0
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def ddp_allreduce_mean(x: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(x)
    t = torch.tensor([float(x)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


# -------------------- data helpers --------------------

def ensure_gene_name(adata: sc.AnnData, gene_col: str = "gene_name") -> sc.AnnData:
    if gene_col not in adata.var.columns:
        adata.var[gene_col] = adata.var_names.astype(str)
    return adata


def ensure_batch_id(adata: sc.AnnData, str_batch: str):
    if "batch_id" in adata.obs.columns:
        return
    if str_batch not in adata.obs.columns:
        raise ValueError(f"Missing obs['{str_batch}'], cannot infer batch_id.")
    adata.obs["batch_id"] = adata.obs[str_batch].astype("category").cat.codes.astype(int)


def get_dense_layer(adata: sc.AnnData, key: str) -> np.ndarray:
    if key not in adata.layers:
        raise KeyError(f"adata.layers missing '{key}'.")
    X = adata.layers[key]
    return X.A if issparse(X) else np.asarray(X)


@torch.no_grad()
def encode_full_dataset_in_chunks(
    raw_model: TransformerModel,
    tokenized_all: dict,
    vocab: GeneVocab,
    batch_labels_np: np.ndarray,
    amp: bool,
    chunk_bs: int,
    device: torch.device,
) -> np.ndarray:
    """
    用 finetune 后的模型对全体细胞推理得到 cell_emb。
    返回 numpy float32: [N, D]
    """
    genes_cpu = tokenized_all["genes"]     # CPU tensor [N,L]
    vals_cpu  = tokenized_all["values"]    # CPU tensor [N,L]
    N = genes_cpu.shape[0]
    pad_id = vocab["<pad>"]

    outs = []
    for s in range(0, N, chunk_bs):
        e = min(N, s + chunk_bs)
        g = genes_cpu[s:e].to(device, non_blocking=True)
        v = vals_cpu[s:e].to(device, non_blocking=True)
        m = g.eq(pad_id)
        b = torch.from_numpy(batch_labels_np[s:e]).long().to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp):
            emb = raw_model.encode_batch(
                src=g,
                values=v.float(),
                src_key_padding_mask=m,
                batch_size=g.size(0),
                batch_labels=b,
                time_step=0,
                return_np=False,
            )

        outs.append(emb.detach().cpu())
        del g, v, m, b, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    emb_all = torch.cat(outs, dim=0).numpy().astype(np.float32)
    return emb_all


# -------------------- core pipeline --------------------

def run_batch_correction_finetune_and_return_emb(
    adata_path: str,
    save_dir: str,
    base_model_dir: str,          # <<< 改这里：分类模型目录
    str_batch: str = "str_batch",
    use_key: str = "X",
    n_hvg: int = 1200,
    n_bins: int = 51,
    is_raw_data: bool = True,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 64,
    dropout: float = 0.2,
    schedule_ratio: float = 0.9,
    schedule_interval: int = 1,
    save_eval_interval: int = 10,
    mask_ratio: float = 0.4,
    ecs_thres: float = 0.8,
    dab_weight: float = 1.0,
    amp: bool = True,
    seed: int = 0,
    test_size: float = 0.1,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    # ---- ddp init ----
    is_ddp, rank, world_size, local_rank = init_distributed_if_needed()
    is_main = (rank == 0)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    logger = scg.logger
    set_seed(seed)

    save_dir = Path(save_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        scg.utils.add_file_handler(logger, save_dir / "run.log")
        logger.info(f"[bc] save_dir={save_dir}")
        logger.info(f"[bc] ddp={is_ddp} world_size={world_size}")

    def emit_progress(message: str, percent: Optional[float] = None, metrics: Optional[Dict[str, Any]] = None) -> None:
        if (not is_main) or (progress_cb is None):
            return
        payload: Dict[str, Any] = {
            "stage": "batch_correction_train",
            "message": message,
        }
        if percent is not None:
            payload["percent"] = float(max(0.0, min(100.0, percent)))
        if metrics is not None:
            payload["metrics"] = metrics
        progress_cb(payload)

    emit_progress("去批次微调准备中", 0.0)

    # ---- load adata ----
    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()
    adata = ensure_gene_name(adata, "gene_name")

    # ---- multi-batch check ----
    if str_batch not in adata.obs.columns:
        if is_main:
            (save_dir / "SKIPPED_NO_BATCH_COLUMN").write_text(f"missing {str_batch}\n", encoding="utf-8")
        return {
            "trained": False,
            "reason": f"missing obs['{str_batch}']",
            "n_batches": 1,
            "cell_emb_bc_npy": None,
            "result_h5ad": None,
            "best_epoch": None,
            "best_val_loss": None,
        }

    n_batches = int(adata.obs[str_batch].nunique())
    if n_batches <= 1:
        if is_main:
            (save_dir / "SKIPPED_SINGLE_BATCH").write_text(f"n_batches={n_batches}\n", encoding="utf-8")
        return {
            "trained": False,
            "reason": "single_batch",
            "n_batches": n_batches,
            "cell_emb_bc_npy": None,
            "result_h5ad": None,
            "best_epoch": None,
            "best_val_loss": None,
        }

    # ---- preprocess: generate X_binned + (batch_id) ----
    emit_progress("去批次微调预处理中", 5.0, {"n_batches": int(n_batches)})
    adata = preprocess_adata(
        adata,
        save_dir=save_dir,
        use_key=use_key,
        ori_batch_col=str_batch,
        n_hvg=n_hvg,
        n_bins=n_bins,
        is_raw_data=is_raw_data,
    )
    adata.layers["X_binned"] = np.asarray(adata.layers["X_binned"])
    ensure_batch_id(adata, str_batch=str_batch)
    num_batch_types = int(adata.obs["batch_id"].nunique())

    # ---- load pretrained configs/vocab/weights ----
    base_model_dir = Path(base_model_dir)
    model_config_file = base_model_dir / "args.json"
    model_file = base_model_dir / "best_model.pt"
    vocab_file = base_model_dir / "vocab.json"

    if not model_config_file.exists():
        raise FileNotFoundError(f"missing {model_config_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"missing {model_file}")
    if not vocab_file.exists():
        raise FileNotFoundError(f"missing {vocab_file}")

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    vocab = GeneVocab.from_file(vocab_file)
    for tok in ["<pad>", "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab["<pad>"])

    # ---- filter genes by vocab ----
    adata.var["id_in_vocab"] = [1 if g in vocab else -1 for g in adata.var["gene_name"].tolist()]
    adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()

    # ---- build arrays ----
    all_counts = get_dense_layer(adata, "X_binned")
    genes = adata.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    batch_ids = adata.obs["batch_id"].to_numpy().astype(int)
    cell_idx = np.arange(adata.n_obs)

    # ---- split train/valid ----
    (train_data, valid_data, train_batch_labels, valid_batch_labels, train_idx, valid_idx) = train_test_split(
        all_counts, batch_ids, cell_idx, test_size=test_size, shuffle=True
    )

    # ---- config for trainer ----
    config = dict(
        seed=seed,
        do_train=True,
        load_model=str(base_model_dir),
        input_layer_key="X_binned",

        # objectives
        GEP=True,
        GEPC=True,
        CLS=False,
        CCE=False,
        ECS=True,
        DAB=True,

        # batch settings
        per_seq_batch_sample=True,
        DSBN=True,
        explicit_zero_prob=True,
        use_batch_labels=True,

        # input
        include_zero_gene=True,
        pre_norm=False,
        amp=amp,

        # your sparsity tricks
        filter0=True,
        kv_mask=True,
        init_cls=False,
        gated_attn=True,

        # hparams
        mask_ratio=mask_ratio,
        epochs=epochs,
        n_bins=n_bins,
        ecs_thres=ecs_thres,
        dab_weight=dab_weight,
        lr=lr,
        batch_size=batch_size,
        dropout=dropout,
        schedule_ratio=schedule_ratio,
        schedule_interval=schedule_interval,
        save_eval_interval=save_eval_interval,
        log_interval = 10,

        # token values from pretrained (important)
        mask_value=model_configs.get("mask_value", -1),
        pad_value=model_configs.get("pad_value", -2),
        cls_value=model_configs.get("cls_value", 0),

        n_hvg=n_hvg,
        max_seq_len=n_hvg + 1,

        pad_token="<pad>",
        input_emb_style=model_configs.get("input_emb_style", "continuous"),
        cell_emb_style=model_configs.get("cell_emb_style", "cls"),
    )
    config = SimpleNamespace(**config)

    if is_main:
        # 仅保存本次训练配置（不保存模型供后续加载，仅做记录）
        with open(save_dir / "train_args.json", "w") as f:
            json.dump(vars(config), f, indent=2)

    # ---- tokenize train/valid ----
    tokenized_train = tokenize_and_pad_batch(
        train_data, gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,
        include_zero_gene=config.include_zero_gene,
        cls_value=config.cls_value,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data, gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,
        include_zero_gene=config.include_zero_gene,
        cls_value=config.cls_value,
    )

    # ---- build model (no CLS head needed; set n_cls=1 placeholder) ----
    ntokens = len(vocab)
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    use_fast_transformer = device.type == "cuda"

    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=4,
        n_cls=1,  # placeholder, not used
        vocab=vocab,
        dropout=config.dropout,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        do_mvc=config.GEPC,
        GEP=config.GEP,
        do_dab=config.DAB,
        use_batch_labels=config.use_batch_labels,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=config.DSBN,
        input_emb_style=config.input_emb_style,
        n_input_bins=config.n_bins,
        cell_emb_style=config.cell_emb_style,
        ecs_threshold=config.ecs_thres,
        explicit_zero_prob=config.explicit_zero_prob,
        pre_norm=config.pre_norm,
        kv_mask=config.kv_mask,
        filter0=config.filter0,
        init_cls=config.init_cls,
        gated_attn=config.gated_attn,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
    )

    # load pretrained weights (shape-matched)
    pretrained = torch.load(model_file, map_location="cpu")
    if isinstance(pretrained, dict) and "state_dict" in pretrained and isinstance(pretrained["state_dict"], dict):
        pretrained = pretrained["state_dict"]
    pretrained = _normalize_scgpt_state_dict_keys(pretrained)
    try:
        model.load_state_dict(pretrained, strict=True)
        if is_main:
            logger.info(f"[bc] loaded pretrained strict=True from {model_file}")
    except Exception as e:
        if is_main:
            logger.warning(f"[bc] strict load failed: {e} | fallback to shape-matched loading.")
        model_dict = model.state_dict()
        filtered = {k: v for k, v in pretrained.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(filtered)
        model.load_state_dict(model_dict, strict=False)

    model = model.to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.schedule_interval, gamma=config.schedule_ratio)
    scaler = torch.amp.GradScaler(enabled=config.amp)
    main_state = SimpleNamespace(optimizer=optimizer, scheduler=scheduler, scaler=scaler)

    # keep "global" batch size consistent
    global_bs = int(config.batch_size)
    per_rank_bs = max(1, global_bs // world_size)

    # ---- training loop ----
    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = -1

    start_time = time.time()

    progress_handler = None
    if is_main and progress_cb is not None:
        for h in list(logger.handlers):
            if getattr(h, "_is_scweb_progress_handler", False):
                logger.removeHandler(h)
        pat = re.compile(r"\| epoch\s+(\d+)\s+\| batch\s+(\d+)/(\d+).*\| loss\s+([0-9.]+)")

        class _ProgressLogHandler(logging.Handler):
            def emit(self, record):
                msg = record.getMessage()
                m = pat.search(msg)
                if not m:
                    return
                ep = int(m.group(1))
                b = int(m.group(2))
                b_all = max(1, int(m.group(3)))
                loss = float(m.group(4))
                pct = 10.0 + ((ep - 1 + (b / b_all)) / max(1, config.epochs)) * 80.0
                emit_progress(
                    "去批次微调训练中",
                    pct,
                    {
                        "epoch": ep,
                        "total_epochs": int(config.epochs),
                        "batch": b,
                        "total_batches": b_all,
                        "loss": loss,
                    },
                )

        progress_handler = _ProgressLogHandler()
        progress_handler.setLevel(logging.INFO)
        progress_handler._is_scweb_progress_handler = True
        logger.addHandler(progress_handler)

    for epoch in range(1, config.epochs + 1):
        emit_progress(
            "去批次微调训练中",
            10.0 + ((epoch - 1) / max(1, config.epochs)) * 80.0,
            {"epoch": int(epoch), "total_epochs": int(config.epochs)},
        )
        # trainer.prepare_data 需要 celltype labels，这里给占位 0
        train_celltype_labels = np.zeros(len(train_batch_labels), dtype=int)
        valid_celltype_labels = np.zeros(len(valid_batch_labels), dtype=int)

        train_data_pt, valid_data_pt = trainer.prepare_data(
            sort_seq_batch=config.per_seq_batch_sample,
            tokenized_train=tokenized_train,
            tokenized_valid=tokenized_valid,
            train_batch_labels=train_batch_labels,
            valid_batch_labels=valid_batch_labels,
            train_celltype_labels=train_celltype_labels,
            valid_celltype_labels=valid_celltype_labels,
            train_idx=train_idx,
            valid_idx=valid_idx,
            config=config,
            epoch=epoch,
        )

        train_loader = trainer.prepare_dataloader(
            train_data_pt, batch_size=per_rank_bs, shuffle=False,
            intra_domain_shuffle=True, drop_last=False, num_workers=0,
            per_seq_batch_sample=config.per_seq_batch_sample,
        )
        valid_loader = trainer.prepare_dataloader(
            valid_data_pt, batch_size=per_rank_bs, shuffle=False,
            intra_domain_shuffle=False, drop_last=False, num_workers=0,
            per_seq_batch_sample=config.per_seq_batch_sample,
        )

        for loader in [train_loader, valid_loader]:
            if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)

        # train
        trainer.train(
            model=model,
            loader=train_loader,
            vocab=vocab,
            criterion_gep_gepc=masked_mse_loss,
            criterion_dab=nn.CrossEntropyLoss(),
            criterion_cls=nn.CrossEntropyLoss(),  # placeholder
            main_state=main_state,
            device=device,
            config=config,
            logger=logger if is_main else None,
            epoch=epoch,
        )

        # eval
        val = trainer.evaluate(
            model=model,
            loader=valid_loader,
            vocab=vocab,
            criterion_gep_gepc=masked_mse_loss,
            criterion_dab=nn.CrossEntropyLoss(),
            criterion_cls=nn.CrossEntropyLoss(),
            device=device,
            config=config,
        )
        val_loss = ddp_allreduce_mean(float(val), device=device)
        emit_progress(
            "去批次微调验证中",
            10.0 + (epoch / max(1, config.epochs)) * 80.0,
            {
                "epoch": int(epoch),
                "total_epochs": int(config.epochs),
                "val_loss": float(val_loss),
                "best_val_loss": float(min(best_val_loss, val_loss)),
            },
        )

        if is_main:
            logger.info(f"[bc] epoch={epoch} valid_loss={val_loss:.4f}")
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                raw_model = model.module if is_ddp else model
                best_state_dict = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
                logger.info(f"[bc] best updated: epoch={best_epoch} loss={best_val_loss:.4f}")

            # 可选：只存记录，不保存模型供后续加载（你不需要）
            if (epoch % config.save_eval_interval == 0) or (epoch == config.epochs):
                with open(save_dir / "train_progress.json", "w") as f:
                    json.dump({"epoch": epoch, "best_epoch": best_epoch, "best_val_loss": best_val_loss}, f, indent=2)

        scheduler.step()
        ddp_barrier()

    # ---- main: use best_state_dict to compute embeddings on full data ----
    cell_emb_path = None
    out_h5ad_path = None

    if is_main:
        if best_state_dict is None:
            raw_model = model.module if is_ddp else model
            best_state_dict = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
            best_epoch = config.epochs
            best_val_loss = float("nan")

        # 重新把最佳权重加载回 raw_model
        raw_model = model.module if is_ddp else model
        raw_model.load_state_dict(best_state_dict, strict=True)
        raw_model.to(device).eval()

        # tokenize full dataset once (CPU)
        tokenized_all = tokenize_and_pad_batch(
            all_counts, gene_ids,
            max_len=config.max_seq_len,
            vocab=vocab,
            pad_token=config.pad_token,
            pad_value=config.pad_value,
            append_cls=True,
            include_zero_gene=config.include_zero_gene,
            cls_value=config.cls_value,
        )

        # 推理得到去批次 embedding
        cell_emb_bc = encode_full_dataset_in_chunks(
            raw_model=raw_model,
            tokenized_all=tokenized_all,
            vocab=vocab,
            batch_labels_np=batch_ids,
            amp=config.amp,
            chunk_bs=config.batch_size,
            device=device,
        )

        # 保存 npy + 写回 h5ad
        cell_emb_path = str(save_dir / "cell_emb_bc.npy")
        np.save(cell_emb_path, cell_emb_bc)

        adata.obsm["X_cell_emb_bc"] = cell_emb_bc
        out_h5ad_path = str(save_dir / "result.h5ad")
        adata.write_h5ad(out_h5ad_path)

        # meta
        meta = {
            "trained": True,
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_batches": int(n_batches),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "elapsed_sec": float(time.time() - start_time),
        }
        with open(save_dir / "train_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"[bc] done. best_epoch={best_epoch} best_val_loss={best_val_loss:.4f}")
        logger.info(f"[bc] saved: {cell_emb_path}")
        logger.info(f"[bc] saved: {out_h5ad_path}")
        emit_progress(
            "去批次微调完成",
            100.0,
            {"best_epoch": int(best_epoch), "best_val_loss": float(best_val_loss)},
        )

    ddp_barrier()

    # cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if progress_handler is not None:
        logger.removeHandler(progress_handler)

    return {
        "trained": True,
        "n_batches": int(n_batches),
        "cell_emb_bc_npy": cell_emb_path,
        "result_h5ad": out_h5ad_path,
        "best_epoch": int(best_epoch) if is_main else None,
        "best_val_loss": float(best_val_loss) if is_main else None,
    }


# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata", required=True, help="input h5ad path")
    p.add_argument("--save_dir", required=True, help="output dir")
    p.add_argument("--base_model_dir", required=True, help="base model dir (args.json/best_model.pt/vocab.json)")

    p.add_argument("--str_batch", default="str_batch")
    p.add_argument("--use_key", default="X")
    p.add_argument("--n_hvg", type=int, default=1200)
    p.add_argument("--n_bins", type=int, default=51)
    p.add_argument("--is_raw_data", action="store_true")
    p.set_defaults(is_raw_data=True)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--schedule_ratio", type=float, default=0.9)
    p.add_argument("--schedule_interval", type=int, default=1)
    p.add_argument("--save_eval_interval", type=int, default=10)

    p.add_argument("--mask_ratio", type=float, default=0.4)
    p.add_argument("--ecs_thres", type=float, default=0.8)
    p.add_argument("--dab_weight", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--payload_json", default="", help="optional path to save payload json on rank0")

    return p.parse_args()


def main():
    args = parse_args()
    payload = run_batch_correction_finetune_and_return_emb(
        adata_path=args.adata,
        save_dir=args.save_dir,
        base_model_dir=args.base_model_dir,
        str_batch=args.str_batch,
        use_key=args.use_key,
        n_hvg=args.n_hvg,
        n_bins=args.n_bins,
        is_raw_data=args.is_raw_data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        dropout=args.dropout,
        schedule_ratio=args.schedule_ratio,
        schedule_interval=args.schedule_interval,
        save_eval_interval=args.save_eval_interval,
        mask_ratio=args.mask_ratio,
        ecs_thres=args.ecs_thres,
        dab_weight=args.dab_weight,
        amp=(not args.no_amp),
        seed=args.seed,
        test_size=args.test_size,
    )
    if args.payload_json:
        is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if is_main:
            Path(args.payload_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
