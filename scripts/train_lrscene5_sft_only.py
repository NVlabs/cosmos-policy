#!/usr/bin/env python
"""Fresh LRSCENE5 SFT-only LoRA control with no mask loss or attention capture."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, default_collate
from megatron.core import parallel_state
from cosmos_policy._src.imaginaire.config import load_config
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.callback import Callback
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init, model_init


def prepare_subset(directory, latest_manifest):
    if distributed.is_rank0():
        cases = [case for case in json.loads(latest_manifest.read_text())['cases']
                 if case['tag'].startswith('LRSCENE5-') and case['tag'] != 'LRSCENE5-init011'
                 and Path(case['source_h5']).is_file()
                 and 'success=True' in Path(case['source_h5']).name]
        if len(cases) != 19:
            raise ValueError(f'Expected 19 LRSCENE5 successful trajectory/mask pairs, got {len(cases)}')
        directory.mkdir(parents=True, exist_ok=True)
        expected = set()
        for case in cases:
            source = Path(case['source_h5']).resolve()
            if not Path(case['mask_npz']).is_file(): raise FileNotFoundError(case['mask_npz'])
            target = directory / source.name
            if not target.exists(): target.symlink_to(source)
            if target.resolve() != source: raise ValueError(f'Trajectory conflict: {target}')
            expected.add(source.name)
        if {path.name for path in directory.glob('*.hdf5')} != expected:
            raise ValueError('Subset directory contains additional or missing trajectories')
        (directory / 'mask_manifest.json').write_text(json.dumps(
            {'total':19, 'complete':19, 'excluded':['LRSCENE5-init011'], 'cases':cases}, indent=2))
    if dist.is_initialized(): dist.barrier()
    return directory / 'mask_manifest.json'


def fixed_batches(dataset):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    batches = []
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            random.seed(20261002); np.random.seed(20261002); torch.manual_seed(20261002)
            for tag, step in [('LRSCENE5-init004',33), ('LRSCENE5-init005',166)]:
                ep = next(i for i,meta in dataset.rollout_episode_metadata.items()
                          if f'--{tag}--' in Path(meta['file_path']).name)
                index = next(i for i,pair in dataset._step_to_episode_map.items() if pair == (ep,step))
                batches.append((tag, step, default_collate([dataset[index]])))
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)
    return batches


class SFTMetrics(Callback):
    def __init__(self, directory, probes, interval=100):
        self.directory = directory
        self.probes = probes
        self.interval = interval
        self.last_lr = None

    def write(self, filename, values):
        if distributed.is_rank0():
            with (self.directory / filename).open('a') as stream:
                stream.write(json.dumps(values, allow_nan=False) + '\n')

    def averaged_metrics(self, output, loss):
        if 'mask_attention_loss' in output or 'mask_attention_map' in output:
            raise RuntimeError('SFT-only control unexpectedly captured mask attention')
        value = loss.detach().float().clone()
        if not torch.isfinite(value): raise RuntimeError('Non-finite SFT-only training loss')
        dist.all_reduce(value); value /= dist.get_world_size()
        return {'sft_loss':value.item(), 'total_loss':value.item(),
                'mask_loss':0.0, 'lambda':0.0, 'use_mask_attention_loss':False}

    def probe(self, model, iteration):
        # Identical augmented images and diffusion noise on every probe.
        # The fixed sample/augmentation/noise settings match the mask-loss run.
        # This uses training samples as an optimization monitor, not held-out evaluation.
        cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        was_training = model.training
        counters = {name:value.clone() for name,value in model.net.named_buffers()
                    if 'accum_' in name}
        rows = []
        try:
            model.eval()
            for offset,(tag,step,batch) in enumerate(self.probes):
                seed = 4100 + offset
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                data = {key:(value.clone().cuda().to(dtype=model.precision)
                             if torch.is_floating_point(value) else value.clone().cuda())
                        if isinstance(value,torch.Tensor) else value for key,value in batch.items()}
                # Match the earlier probes; no backward or optimizer update here.
                with torch.enable_grad():
                    output, loss = model.training_step(data, iteration)
                    rows.append({'tag':tag, 'relative_step':step, **self.averaged_metrics(output,loss)})
                del data, output, loss
        finally:
            with torch.no_grad():
                for name,value in model.net.named_buffers():
                    if name in counters: value.copy_(counters[name])
            model.train(was_training)
            torch.set_rng_state(cpu_state); torch.cuda.set_rng_state(cuda_state)
            random.setstate(python_state); np.random.set_state(numpy_state)
        means = {key:sum(row[key] for row in rows)/len(rows)
                 for key in ['sft_loss','mask_loss','total_loss','lambda']}
        record = {'iteration':iteration,'kind':'fixed_training_probe','samples':rows,**means}
        self.write('fixed_probe_metrics.jsonl', record)
        if distributed.is_rank0(): print('[SFT_FIXED_PROBE] '+json.dumps(record), flush=True)

    def on_train_start(self, model, iteration=0):
        if model.config.use_mask_attention_loss:
            raise ValueError('This control requires use_mask_attention_loss=False')
        if any('agentview_mask' in batch for _,_,batch in self.probes):
            raise ValueError('SFT-only dataset must not load mask supervision')
        self.probe(model, iteration)

    def on_before_optimizer_step(self, model_ddp, optimizer, scheduler, grad_scaler, iteration=0):
        self.last_lr = optimizer.param_groups[0]['lr']

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        record = {'iteration':iteration,'kind':'training','learning_rate':self.last_lr,
                  **self.averaged_metrics(output_batch,loss)}
        self.write('train_metrics.jsonl', record)
        if distributed.is_rank0() and (iteration <= 2 or iteration % 10 == 0):
            print('[SFT_TRAIN] '+json.dumps(record), flush=True)
        if iteration % self.interval == 0 or iteration == self.config.trainer.max_iter:
            self.probe(model, iteration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-name', required=True)
    parser.add_argument('--max-iter', type=int, default=2742)
    parser.add_argument('--probe-every', type=int, default=100)
    args = parser.parse_args()
    if args.max_iter < 1 or args.probe_every < 1: raise ValueError('Iteration counts must be positive')
    run_dir = REPO / 'scripts/experiments/lrscene5_sft_only' / args.job_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = REPO / 'training/tta_sft_metadata'
    subset = REPO / 'training/tta_sft_lrscene5_mask_19'
    os.environ['TTA_REPAIR_SFT_METADATA_DIR'] = str(metadata)
    os.environ['TTA_REPAIR_SFT_ROLLOUT_DIR'] = str(subset)
    os.environ.pop('TTA_REPAIR_SFT_DROP_TAIL',None)
    os.environ.pop('DETERMINISTIC',None)
    os.environ.setdefault('IMAGINAIRE_OUTPUT_ROOT', str(REPO.parents[1] / 'checkpoints'))
    with distributed_init(): distributed.init()
    manifest = prepare_subset(subset, REPO/'experiments/tta/mask_view/latest/manifest.json')
    spec = importlib.util.spec_from_file_location('archived_fixed_sft',
        REPO/'tests/tta/archived_lorafix/train_tta_sft_fixed.py')
    fixed = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixed)
    config = load_config('cosmos_policy/config/config.py',
                        ['--','experiment=tta_repair_sft'], enable_one_logger=False)
    fixed.mutate_config(config, SimpleNamespace(max_iter=args.max_iter, job_name=args.job_name))
    config.model.config.fsdp_shard_size = dist.get_world_size()
    config.model.config.use_mask_attention_loss = False
    config.model.config.mask_attention_lambda = 1.0
    config.model.config.mask_attention_layers = 4
    config.dataloader_train.dataset.agentview_mask_manifest = ''
    config.validate(); config.freeze()
    if distributed.is_rank0() and Path(config.job.path_local,'checkpoints/latest_checkpoint.txt').exists():
        raise ValueError('Use a new job name: this run must start with a fresh LoRA')
    trainer = config.trainer.type(config)
    with model_init(): model = instantiate(config.model)
    n_lora = fixed.reinit_lora(model.net)
    dataset = instantiate(config.dataloader_train.dataset)
    if dataset.num_episodes != 19 or dataset.num_steps != 4500:
        raise ValueError(f'Unexpected dataset: {dataset.num_episodes} episodes, {dataset.num_steps} steps')
    probes = fixed_batches(dataset)
    monitor = SFTMetrics(run_dir, probes, args.probe_every)
    monitor.config, monitor.trainer = config, trainer
    trainer.callbacks._callbacks.append(monitor)
    sampler = DistributedSampler(dataset, num_replicas=parallel_state.get_data_parallel_world_size(),
                                 rank=parallel_state.get_data_parallel_rank(), shuffle=True, seed=0)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, drop_last=True,
                           num_workers=0, persistent_workers=False, pin_memory=True)
    if distributed.is_rank0():
        record = {'job_name':args.job_name, 'world_size':dist.get_world_size(), 'batch_size_per_gpu':1,
                  'global_batch_size':dist.get_world_size(), 'max_iter':args.max_iter, 'learning_rate':1e-5,
                  'lambda':0.0, 'use_mask_attention_loss':False,
                  'attention_capture_enabled':False, 'lora_rank':8, 'lora_alpha':16,
                  'fresh_lora_modules':n_lora, 'trajectories':19, 'steps':4500,
                  'excluded':['LRSCENE5-init011'], 'data_directory':str(subset),
                  'source_manifest':str(manifest), 'mask_manifest':'',
                  'base_checkpoint':config.checkpoint.load_path,
                  'checkpoint_directory':str(Path(config.job.path_local)/'checkpoints'),
                  'fixed_probe_every':args.probe_every, 'started_at':time.strftime('%Y-%m-%d %H:%M:%S')}
        (run_dir/'run.json').write_text(json.dumps(record,indent=2))
        print('[SFT_RUN] '+json.dumps(record), flush=True)
    trainer.train(model, dataloader, None)
    if distributed.is_rank0(): (run_dir/'completed.json').write_text(json.dumps({'iteration':args.max_iter}))
    dist.destroy_process_group()


if __name__ == '__main__': main()
