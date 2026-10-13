"""Precompute T5 embeddings for LIBERO-PRO suite instructions (一次性预处理).

背景
    run_libero_eval 启动时只加载预计算的 T5 embedding 缓存；一旦遇到缓存里没有的
    指令，get_t5_embedding_from_cache() 会**现场加载 T5 编码器**（text_encoder/
    约 9GB），与策略模型同时驻留极易 OOM。
    LIBERO-PRO 的 _task / _lan 变体用的是全新指令，全部不在缓存里（缓存现有 63 条）。

做法
    本脚本**只加载 T5，不加载策略**，把指定 suite 的全部指令编码后合并进
    libero_t5_embeddings.pkl。原文件自动备份为 .backup。
    跑完之后 eval 走正常加载路径即可，不会再触发 T5。

用法
    cd <repo>
    export LIBERO_CONFIG_PATH=<LIBERO-PRO>/configs/libero_pro
    export PYTHONPATH=<LIBERO-PRO>:$PWD
    export HF_HUB_OFFLINE=1
    python scripts/libero_pro/precompute_t5.py --suite libero_10_task --dry-run   # 先看缺哪些
    python scripts/libero_pro/precompute_t5.py --suite libero_10_task             # 真正编码
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]

DEFAULT_CACHE = (
    "/data1/liu/exp/counterfactual/checkpoints/"
    "Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl"
)


def collect_instructions(suites: list[str]) -> dict[str, tuple[str, str]]:
    """返回 {instruction: (suite, task_name)}，用 benchmark 保证与 eval 完全一致。"""
    from libero.libero import benchmark

    targets: dict[str, tuple[str, str]] = {}
    for suite in suites:
        bench = benchmark.get_benchmark_dict()[suite]()
        names = bench.get_task_names()
        for index in range(len(names)):
            task = bench.get_task(index)
            targets[str(task.language)] = (suite, str(names[index]))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="可重复；默认 libero_10_task",
    )
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="T5 embedding 缓存路径")
    parser.add_argument("--device", default="cuda:0", help="T5 编码设备；显存紧张可用 cpu")
    parser.add_argument("--dry-run", action="store_true", help="只列出缺失，不加载 T5")
    args = parser.parse_args()

    suites = args.suite or ["libero_10_task"]
    targets = collect_instructions(suites)
    print(f"suite={suites} 共 {len(targets)} 条指令")

    cache_path = Path(args.cache)
    if not cache_path.exists():
        parser.error(f"cache not found: {cache_path}")
    with open(cache_path, "rb") as handle:
        cache = pickle.load(handle)
    print(f"现有缓存 {len(cache)} 条")

    missing = [text for text in targets if text not in cache]
    print(f"缺失 {len(missing)} 条:")
    for text in missing:
        suite, task_name = targets[text]
        print(f"   [{suite}] {text!r}")
    for text in targets:
        if text in cache:
            print(f"   已有   {text!r}")
    if not missing:
        print("无需计算，缓存已完整。")
        return
    if args.dry_run:
        print("--dry-run：未加载 T5，未写盘。")
        return

    from cosmos_policy._src.predict2.inference.get_t5_emb import get_text_embedding

    for text in missing:
        embedding = get_text_embedding(text, device=args.device)
        if not hasattr(embedding, "detach"):
            parser.error(f"unexpected embedding type for {text!r}: {type(embedding)!r}")
        cache[text] = embedding.detach().cpu()
        print(f"   encoded {text!r} -> {tuple(cache[text].shape)}")

    # 不用固定的 .backup：它可能已被之前的运行占用，直接写会覆盖掉。
    backup_path = Path(str(cache_path) + ".backup")
    if backup_path.exists():
        stamp = __import__("time").strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"{cache_path}.backup_{stamp}")
    shutil.copy2(cache_path, backup_path)
    print(f"已备份原缓存 -> {backup_path}")
    with open(cache_path, "wb") as handle:
        pickle.dump(cache, handle)
    print(f"已写回 {cache_path}（共 {len(cache)} 条）")


if __name__ == "__main__":
    main()
