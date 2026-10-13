#!/usr/bin/env python
"""修复版 TTA SFT 启动器。

根因（2026-09-13 探针链确认）：
  text2world_model.build_net() 用 meta device 构建 → net.to_empty("cuda") 物化
  → net.init_weights() 只初始化基座、不认识 PEFT 注入的 LoRA → 全部 LoRA 以
  A=B=0 出生 → grad_A=Bᵀ·up=0 且 grad_B=up⊗A(x)=0，永久死锁。
  少数层（blocks 0-3）因 to_empty 分配到非零垃圾内存而逃逸并过拟合爆炸。

本启动器在模型实例化后重新初始化 LoRA（A=kaiming_uniform(a=√5)、B=0，PEFT
标准），并默认把 optimizer 从自研 FusedAdam 换成 torch 标准 AdamW。其余流程
与 cosmos_policy/scripts/train.py 的 launch() 完全一致。

用法（单 GPU，torchrun 1 进程）：
  IMAGINAIRE_OUTPUT_ROOT=... \
  torchrun --nproc_per_node=1 tests/tta/train_tta_sft_fixed.py \
    --job-name tta_repair_sft_ks4_fix --max-iter 800
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from loguru import logger as logging
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

REPO = Path(__file__).resolve().parents[2]

from cosmos_policy._src.imaginaire.config import Config, load_config  # noqa: E402
from cosmos_policy._src.imaginaire.lazy_config import instantiate  # noqa: E402
from cosmos_policy._src.imaginaire.utils import distributed  # noqa: E402
from cosmos_policy._src.imaginaire.utils.context_managers import (  # noqa: E402
    data_loader_init,
    distributed_init,
    model_init,
)
from cosmos_policy._src.imaginaire.utils.launch import log_reproducible_setup  # noqa: E402


def reinit_lora(net: torch.nn.Module) -> int:
    """PEFT 标准初始化：A=kaiming_uniform(a=√5)，B=0。返回处理的层数。"""
    n = 0
    for name, mod in net.named_modules():
        if name.endswith("lora_A.default"):
            torch.nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))
            n += 1
        elif name.endswith("lora_B.default"):
            torch.nn.init.zeros_(mod.weight)
    return n


def mutate_config(config: Config, args) -> None:
    """在 freeze 前做训练配置变更。"""
    # 数据与训练长度
    opts = [
        ("trainer.max_iter", args.max_iter),
        ("scheduler.cycle_lengths", [args.max_iter, 100000000000000]),
        ("scheduler.warm_up_steps", [25, 0]),
        ("checkpoint.save_iter", 400),
        ("job.name", args.job_name),
    ]
    for key, val in opts:
        node = config
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p] if isinstance(node, dict) else getattr(node, p)
        last = parts[-1]
        if isinstance(node, dict):
            node[last] = val
        else:
            setattr(node, last, val)

    # 换标准 AdamW（去掉 fusedadam 专有参数）
    opt = config.optimizer
    try:
        opt.optim_type = "adamw"
        if hasattr(opt, "master_weights"):
            del opt.master_weights
        if hasattr(opt, "capturable"):
            opt.capturable = False
        logging.info("optimizer 换为 torch 标准 AdamW")
    except Exception as exc:
        logging.warning(f"optimizer 切换失败（保留 fusedadamw）: {exc}")

    # 单进程：关掉 worker 子进程
    dl = config.dataloader_train
    try:
        dl.num_workers = 0
        dl.persistent_workers = False
    except Exception as exc:
        logging.warning(f"dataloader 变更失败: {exc}")


def launch(config: Config, args) -> None:
    args.config = "cosmos_policy/config/config.py"  # log_reproducible_setup 需要
    args.opts = []
    with distributed_init():
        distributed.init()

    config.validate()
    config.freeze()
    trainer = config.trainer.type(config)
    log_reproducible_setup(config, args)

    with model_init():
        model = instantiate(config.model)

    # === 修复点：物化之后重新初始化 LoRA ===
    n_lora = reinit_lora(model.net)
    logging.info(f"[FIX] LoRA 重新初始化完成: {n_lora} 层 (A=kaiming, B=0)")
    a_mod = next(m for n, m in model.net.named_modules() if n.endswith("lora_A.default"))
    b_mod = next(m for n, m in model.net.named_modules() if n.endswith("lora_B.default"))
    logging.info(f"[FIX] 抽查: A norm={a_mod.weight.float().norm().item():.4f} "
                 f"B norm={b_mod.weight.float().norm().item():.4f}")

    with data_loader_init():
        dataset = instantiate(config.dataloader_train.dataset)
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=parallel_state.get_data_parallel_world_size(),
            rank=parallel_state.get_data_parallel_rank(),
            shuffle=True,
            seed=0,
        )
        dataloader_train = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=config.dataloader_train.batch_size,
            drop_last=config.dataloader_train.drop_last,
            num_workers=config.dataloader_train.num_workers,
            persistent_workers=config.dataloader_train.persistent_workers,
            pin_memory=config.dataloader_train.pin_memory,
            pin_memory_device=config.dataloader_train.pin_memory_device,
            timeout=config.dataloader_train.timeout,
        )
        dataloader_val = None

    trainer.train(model, dataloader_train, dataloader_val)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-name", default="tta_repair_sft_ks4_fix")
    ap.add_argument("--max-iter", type=int, default=800)
    ap.add_argument("--data-dir", default=str(REPO / "training/tta_sft_ks4_only"))
    ap.add_argument("--metadata-dir", default=str(REPO / "training/tta_sft_metadata"))
    args = ap.parse_args()

    os.environ.setdefault(
        "IMAGINAIRE_OUTPUT_ROOT", "/data1/liu/exp/counterfactual/checkpoints"
    )
    os.environ.setdefault(
        "TTA_REPAIR_SFT_METADATA_DIR", str(REPO / "training/tta_sft_metadata")
    )
    os.environ.setdefault("TTA_REPAIR_SFT_ROLLOUT_DIR", args.data_dir)

    config = load_config(
        "cosmos_policy/config/config.py",
        ["--", "experiment=tta_repair_sft"],
        enable_one_logger=True,
    )
    mutate_config(config, args)
    launch(config, args)


if __name__ == "__main__":
    main()
