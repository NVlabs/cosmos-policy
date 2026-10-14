#!/usr/bin/env python3
"""Generate extra Cosmos T5 embeddings for LIBERO-plus language perturbations."""

import argparse
import pathlib
import pickle
import re

import torch
from libero.libero import benchmark
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
from tqdm import tqdm

from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder


DEFAULT_BASE_TASK = "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
DEFAULT_LANGUAGE_TASK = f"{DEFAULT_BASE_TASK}_language_5_view_0_0_100_0_0_initstate_0"
MAIN_SUITES = ["libero_spatial", "libero_object", "libero_goal"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute exact T5 embeddings for LIBERO-plus language perturbation instructions."
    )
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--category", default="Language Instructions")
    parser.add_argument("--base-task", default=DEFAULT_BASE_TASK)
    parser.add_argument(
        "--all-base-tasks",
        action="store_true",
        help="Generate embeddings for every language perturbation variant for every base task in the selected suite.",
    )
    parser.add_argument(
        "--all-main-suites",
        action="store_true",
        help="Generate embeddings for libero_spatial, libero_object, and libero_goal.",
    )
    parser.add_argument(
        "--task-name",
        action="append",
        default=None,
        help="Exact LIBERO-plus task name. Can be passed multiple times. Defaults to the smoke-test language_5 task.",
    )
    parser.add_argument(
        "--all-language-variants",
        action="store_true",
        help="Generate embeddings for every language perturbation variant matching --base-task.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-path",
        default="/data3/liu/exp/counterfactual/checkpoints/t5-11b",
        help="Local google-t5/t5-11b directory or HuggingFace model name.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=True,
        help="Load the T5 model/tokenizer from local files only.",
    )
    parser.add_argument(
        "--output",
        default="./experiments/paired_smoke/libero_plus_language_t5_embeddings.pkl",
    )
    return parser.parse_args()


def strip_perturbation_suffix(task_name):
    """Remove LIBERO-plus perturbation suffixes to recover the clean base task name."""
    for sep in ["_language_", "_light_", "_table_", "_tb_"]:
        if sep in task_name:
            return task_name.split(sep)[0]
    if "_view_" in task_name:
        return task_name.split("_view_")[0]
    match = re.match(r"(.+)_add_\d+$", task_name)
    if match:
        return match.group(1)
    match = re.match(r"(.+)_noise_\d+$", task_name)
    if match:
        return match.group(1)
    match = re.match(r"(.+)_level\d_sample\d$", task_name)
    if match:
        return match.group(1)
    return task_name


def get_base_tasks(suite_name):
    seen = set()
    base_tasks = []
    for task_name in libero_task_map[suite_name]:
        base_task = strip_perturbation_suffix(task_name)
        if base_task not in seen:
            seen.add(base_task)
            base_tasks.append(base_task)
    return base_tasks


def get_task_languages(suite_name, category, base_task, task_names, all_language_variants, all_base_tasks):
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"Unknown LIBERO suite: {suite_name}")

    task_suite = benchmark_dict[suite_name](category_value=category)
    languages = []

    if all_base_tasks:
        base_tasks = set(get_base_tasks(suite_name))
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            if strip_perturbation_suffix(task.name) in base_tasks:
                languages.append(task.language)
        if not languages:
            raise RuntimeError(f"No language variants found for suite: {suite_name}")
        return languages

    if all_language_variants:
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            if strip_perturbation_suffix(task.name) == base_task:
                languages.append(task.language)
        if not languages:
            raise RuntimeError(f"No language variants found for base task: {base_task}")
        return languages

    selected_task_names = task_names or [DEFAULT_LANGUAGE_TASK]
    for task_name in selected_task_names:
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            if task.name == task_name:
                languages.append(task.language)
                break
        else:
            raise RuntimeError(f"Task not found in suite={suite_name}, category={category}: {task_name}")

    return languages


def main():
    args = parse_args()
    if args.all_main_suites and not args.all_base_tasks:
        raise ValueError("--all-main-suites should be used with --all-base-tasks")
    if args.all_base_tasks and args.task_name:
        raise ValueError("--task-name cannot be combined with --all-base-tasks")

    suites = MAIN_SUITES if args.all_main_suites else [args.suite]
    all_languages = []
    for suite_name in suites:
        suite_languages = get_task_languages(
            suite_name,
            args.category,
            args.base_task,
            args.task_name,
            args.all_language_variants,
            args.all_base_tasks,
        )
        print(f"{suite_name}: {len(suite_languages)} language instruction(s)")
        all_languages.extend(suite_languages)

    languages = list(dict.fromkeys(all_languages))

    print(f"Generating {len(languages)} T5 embedding(s):")
    for language in languages:
        print(f"  - {language}")

    encoder = CosmosT5TextEncoder(
        model_name=args.model_path,
        device=args.device,
        local_files_only=args.local_files_only,
        use_safetensors=False,
    )

    embeddings = {}
    for language in tqdm(languages):
        embeddings[language] = encoder.encode_prompts(language).to(dtype=torch.bfloat16).cpu()

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(embeddings, file)
    print(f"Saved extra T5 embeddings to: {output_path}")


if __name__ == "__main__":
    main()
