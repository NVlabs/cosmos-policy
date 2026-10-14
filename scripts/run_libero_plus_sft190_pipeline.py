#!/usr/bin/env python3
"""Train fresh LoRA or full-parameter SFT, export, and evaluate original cases."""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import run_libero_plus_batch as batch
from run_lrscene5_sft_only_pipeline import verify_checkpoint, verify_merge

PYTHON = sys.executable
BASE = REPO.parents[1] / 'checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt'
SELECTION = REPO / 'experiments/tta/repair_libero_plus_435/success_task_p95/selected_manifest.json'
DATA = REPO / 'training/tta_sft_libero_plus_190_p95'
BASELINE = REPO / 'experiments/libero_plus/cosmospolicy_full_7dims'


def read_results(directory, expected_task_ids=None):
    results = {}
    for path in directory.glob('*/episodes/task*/result.json'):
        row = batch.read_json(path)
        if row['task_id'] in results:
            raise ValueError(f"Duplicate evaluated task: {row['task_id']}")
        results[row['task_id']] = row
    expected = set(range(2519)) if expected_task_ids is None else set(expected_task_ids)
    if set(results) != expected or any(row['status'] != 'completed' for row in results.values()):
        raise ValueError(f'Evaluation incomplete: {len(results)} results')
    return results


def evaluation_cases(selection, scope):
    old = read_results(BASELINE)
    trained = {row['task_id'] for row in selection['cases']}
    if scope == 'all':
        cases = list(old.values())
    elif scope == 'training-cases':
        subset = selection.get('evaluation_subset', {})
        task_ids = subset.get('task_ids', [])
        if (subset.get('scope') != scope or subset.get('num_cases') != len(trained)
                or len(task_ids) != len(trained) or set(task_ids) != trained):
            raise ValueError('Training-case evaluation must exactly match the immutable training selection')
        if any(task_id not in old for task_id in task_ids):
            raise ValueError('Training-case evaluation must use recorded baseline case identities')
        for row in selection['cases']:
            previous = old[row['task_id']]
            if row['task_name'] != previous['task_name'] or row['group'] != previous['group']:
                raise ValueError('Training-case identity differs from the original baseline case')
        cases = [old[task_id] for task_id in task_ids]
    elif scope == 'camera-memory-failures':
        subset = selection.get('evaluation_subset', {})
        task_ids = subset.get('task_ids', [])
        if (subset.get('scope') != scope or not task_ids
                or subset.get('num_cases') != len(task_ids)
                or any(type(task_id) is not int for task_id in task_ids)
                or len(set(task_ids)) != len(task_ids)):
            raise ValueError('Require an immutable list of camera Memory failures in the manifest')
        if (set(task_ids).intersection(trained)
                or any(task_id not in old or old[task_id]['group'] != 'camera_viewpoints'
                       or old[task_id]['success'] is not False for task_id in task_ids)):
            raise ValueError('Camera Memory test cases must be baseline failures outside the training set')
        cases = [old[task_id] for task_id in task_ids]
    else:
        failure_group = {
            'robotinit-failures': 'robot_initial_states',
            'robotinit-heldout-failures': 'robot_initial_states',
            'camera-failures': 'camera_viewpoints',
            'background-heldout-failures': 'background_textures',
            'sensor-noise-failures': 'sensor_noise',
        }[scope]
        cases = [row for row in old.values()
                 if row['group'] == failure_group and row['success'] is False
                 and (scope not in {'robotinit-heldout-failures', 'background-heldout-failures'}
                      or row['task_id'] not in trained)]
        if scope in {'background-heldout-failures', 'sensor-noise-failures'}:
            subset = selection.get('evaluation_subset', {})
            task_ids = subset.get('task_ids', [])
            expected = {row['task_id'] for row in cases}
            if (subset.get('scope') != scope or subset.get('num_cases') != len(expected)
                    or len(task_ids) != len(expected) or set(task_ids) != expected):
                raise ValueError('Evaluation differs from the immutable manifest subset')
    return [dict(task_id=row['task_id'], task_name=row['task_name'], group=row['group'],
                 baseline_success=row['success'], in_training=row['task_id'] in trained)
            for row in sorted(cases, key=lambda row:row['task_id'])]


def compare_results(evaluation, selection, checkpoint, scope='all'):
    planned = evaluation_cases(selection, scope)
    old, new = read_results(BASELINE), read_results(evaluation, [row['task_id'] for row in planned])
    old_config = batch.read_json(BASELINE / 'run_config.json')
    new_config = batch.read_json(evaluation / 'run_config.json')
    for key in ['manifest_index_sha256', 'source_sha256', 'action_seed', 'environment_seed',
                'settle_steps', 'max_steps', 'rng_mode', 'num_trials_per_task', 'init_state_index']:
        if old_config[key] != new_config[key]:
            raise ValueError(f'Evaluation protocol mismatch: {key}')
    trained = {row['task_id'] for row in selection['cases']}
    if len(trained) != selection['num_cases']:
        raise ValueError('The selected training cases must be distinct recorded baseline cases')
    trajectory_sources = collections.Counter()
    for row in selection['cases']:
        source = row.get('trajectory_source', 'memory_repair')
        if source not in {'memory_repair', 'baseline_success'}:
            raise ValueError(f'Unknown trajectory source: {source}')
        if old[row['task_id']]['success'] is not (source == 'baseline_success'):
            raise ValueError(f"Training source disagrees with baseline outcome: {row['task_id']}")
        trajectory_sources[source] += 1
    partitions = collections.defaultdict(list)
    categories = collections.defaultdict(list)
    for task_id, row in new.items():
        if row['task_name'] != old[task_id]['task_name']:
            raise ValueError(f'Task identity mismatch: {task_id}')
        for artifact in [*row['videos'], row['actions_path'], row['episode_hdf5']]:
            if not Path(artifact).is_file():
                raise ValueError(f'Missing evaluation artifact: {artifact}')
        category = ('training_cases' if task_id in trained else
                    'heldout_baseline_failures' if not old[task_id]['success'] else
                    'baseline_successes')
        partitions[category].append(row)
        categories[row['group']].append(row)
    def describe(rows):
        successes = sum(row['success'] is True for row in rows)
        before = sum(old[row['task_id']]['success'] is True for row in rows)
        return dict(total=len(rows), baseline_successes=before, successes=successes,
                    success_rate=successes/len(rows),
                    newly_successful=sum(not old[row['task_id']]['success'] and row['success'] for row in rows),
                    regressions=sum(old[row['task_id']]['success'] and not row['success'] for row in rows))
    return dict(status='COMPLETED', checkpoint=str(checkpoint),
                scope='SFT checkpoint alone, without Memory intervention; original case identities and seeds',
                evaluation_scope=scope, evaluation_cases=len(planned),
                training_trajectories=len(trained), protocol={key:new_config[key] for key in
                    ['action_seed','environment_seed','settle_steps','max_steps','rng_mode','num_trials_per_task','init_state_index']},
                trajectory_sources=dict(trajectory_sources),
                evaluated_training_cases=len(trained.intersection(new)),
                training_cases_outside_evaluation=len(trained.difference(new)),
                evaluation_subset_reference=selection.get('evaluation_subset')
                    if scope in {'training-cases', 'camera-memory-failures', 'background-heldout-failures', 'sensor-noise-failures'} else None,
                overall=describe(list(new.values())),
                by_partition={key:describe(rows) for key,rows in sorted(partitions.items())},
                by_category={key:describe(rows) for key,rows in sorted(categories.items())},
                evaluation_directory=str(evaluation), baseline_directory=str(BASELINE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-name', required=True)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--epochs', type=int, default=1)
    duration.add_argument('--max-iter', type=int)
    parser.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
    parser.add_argument('--manifest', type=Path, default=SELECTION)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--expected-cases', type=int, default=190)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--full-finetune', action='store_true',
                        help='Train the full policy backbone without LoRA')
    parser.add_argument('--batch-size-per-gpu', type=int, default=1)
    parser.add_argument('--eval-epochs', type=int, nargs='+',
                        help='Save epoch checkpoints and evaluate these epochs after training')
    parser.add_argument('--evaluation-scope', choices=['all','training-cases','robotinit-failures','robotinit-heldout-failures','camera-failures','camera-memory-failures','background-heldout-failures','sensor-noise-failures'], default='all')
    parser.add_argument('--continue-training-pid', type=int,
                        help='Replace the supervisor while keeping an existing torchrun training process')
    args = parser.parse_args()
    if not args.full_finetune and (args.lora_rank < 1 or args.lora_alpha < 1):
        parser.error('LoRA rank and alpha must be positive')
    if args.batch_size_per_gpu < 1:
        parser.error('Batch size per GPU must be positive')
    if args.expected_cases < 1 or args.epochs < 1 or (args.max_iter is not None and args.max_iter < 3) or len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error('Require at least three full-training iterations and distinct nonnegative GPU IDs')
    if args.eval_epochs is not None:
        if (args.max_iter is not None or len(set(args.eval_epochs)) != len(args.eval_epochs)
                or any(epoch < 1 or epoch > args.epochs for epoch in args.eval_epochs)
                or args.epochs not in args.eval_epochs):
            parser.error('Evaluation epochs must be distinct, within --epochs, and include the final epoch')
        args.eval_epochs = sorted(args.eval_epochs)
    evaluation_epochs = args.eval_epochs or ([args.epochs] if args.max_iter is None else [None])
    run = REPO / 'scripts/experiments/libero_plus_sft190' / args.job_name
    if run.exists() and any(run.iterdir()) and args.continue_training_pid is None:
        raise ValueError('Use a new job name to preserve previous runs')
    run.mkdir(parents=True, exist_ok=args.continue_training_pid is not None)
    previous_state = batch.read_json(run/'pipeline_status.json') if args.continue_training_pid is not None else {}
    if previous_state and (previous_state['status'] != 'TRAINING' or previous_state['job_name'] != args.job_name
                           or previous_state['gpu_ids'] != args.gpus):
        raise ValueError('Only the currently running training job can have its supervisor replaced')
    selected = batch.read_json(args.manifest)
    if (selected['num_cases'] != args.expected_cases or len(selected['cases']) != args.expected_cases
            or len({row['task_id'] for row in selected['cases']}) != args.expected_cases):
        raise ValueError(f'Expected {args.expected_cases} distinct selected training cases')
    expected_sources = dict(collections.Counter(row.get('trajectory_source', 'memory_repair')
                                               for row in selected['cases']))
    planned = evaluation_cases(selected, args.evaluation_scope)
    selection_path = run/'evaluation_cases.json'
    batch.atomic_json(selection_path, dict(evaluation_scope=args.evaluation_scope,
                      task_ids=[row['task_id'] for row in planned], cases=planned))
    state = dict(previous_state)
    state.update(job_name=args.job_name, pid=os.getpid(), run_directory=str(run),
                 training_trajectories=args.expected_cases, evaluation_cases=len(planned),
                 trajectory_sources=expected_sources,
                 max_iter=previous_state.get('max_iter', args.max_iter),
                 epochs=args.epochs if args.max_iter is None else None,
                 gpu_ids=args.gpus, use_mask_attention_loss=False, base_checkpoint=str(BASE),
                 full_finetune=args.full_finetune, use_lora=not args.full_finetune,
                 training_mode='full' if args.full_finetune else 'lora',
                 lora_rank=None if args.full_finetune else args.lora_rank,
                 lora_alpha=None if args.full_finetune else args.lora_alpha,
                 batch_size_per_gpu=args.batch_size_per_gpu,
                 global_batch_size=args.batch_size_per_gpu*len(args.gpus),
                 source_manifest=str(args.manifest), evaluation_scope=args.evaluation_scope,
                 evaluation_epochs=evaluation_epochs, evaluated_epochs=[],
                 evaluation_selection=str(selection_path),
                 started_at=previous_state.get('started_at', time.strftime('%Y-%m-%d %H:%M:%S')),
                 supervisor_paused_for_evaluation_scope_change=False)
    def update(status, **extra):
        state.update(status=status, updated_at=time.strftime('%Y-%m-%d %H:%M:%S'), **extra)
        batch.atomic_json(run / 'pipeline_status.json', state)
        batch.atomic_json(Path('/tmp/libero_plus_sft190_latest_run.json'), state)
        print('[SFT190_PIPELINE] ' + json.dumps(state), flush=True)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('COSMOS_TTA_', 'TTA_REPAIR_')):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,args.gpus)), HF_HUB_OFFLINE='1',
               HF_HUB_CACHE='/tmp/tta_hf_cache/huggingface-hub', OMP_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               TTA_REPAIR_SFT_BASE_CHECKPOINT=str(BASE), COSMOS_INITIAL_ALIGNMENT='0',
               COSMOS_SKILL_COMPLETION_SHADOW='0', COSMOS_SKILL_COMPLETION_ACTIVE='0',
               COSMOS_SKILL_READY_MEMORY='0', COSMOS_CUROBO_JOINT_EXECUTION='0')
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH','')
    train_script = REPO / 'scripts/train_libero_plus_sft190.py'
    def verify_training_mode(recipe):
        if (recipe.get('full_finetune', False) != args.full_finetune
                or recipe['base_checkpoint'] != str(BASE) or recipe['use_mask_attention_loss']):
            raise ValueError('Training mode or initialization differs from the requested recipe')
        if args.full_finetune:
            if (recipe['fresh_lora_modules'] != 0 or recipe.get('use_lora') is not False
                    or recipe.get('policy_parameters', 0) < 1
                    or recipe['trainable_parameters'] != recipe['policy_parameters']):
                raise ValueError('Full-parameter training must update all policy weights without adapters')
        elif (recipe['fresh_lora_modules'] != 280
                or recipe['lora_rank'] != args.lora_rank or recipe['lora_alpha'] != args.lora_alpha
                or recipe['lora_target_modules'] != ['q_proj','k_proj','v_proj','output_proj','mlp.layer1','mlp.layer2']):
            raise ValueError('Training LoRA configuration differs from the requested recipe')

    def export_checkpoint(checkpoint_dir, output, log_path):
        export_env = dict(env, CUDA_VISIBLE_DEVICES=str(args.gpus[0]))
        if args.full_finetune:
            command = [PYTHON, str(REPO/'scripts/export_full_finetune_ckpt.py'),
                       '--ckpt', str(checkpoint_dir/'model'), '--out', str(output)]
        else:
            command = [PYTHON, str(REPO/'scripts/merge_lora_ckpt.py'),
                       '--ckpt', str(checkpoint_dir/'model'), '--out', str(output),
                       '--lora-rank', str(args.lora_rank), '--lora-alpha', str(args.lora_alpha)]
        with log_path.open('w') as log:
            subprocess.run(command, cwd=REPO, env=export_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return verify_merge(output)

    def verify_training(job_name, iterations):
        directory = REPO / 'scripts/experiments/libero_plus_sft190' / job_name
        completed = batch.read_json(directory / 'completed.json')
        recipe = batch.read_json(directory / 'run.json')
        if iterations is None:
            iterations = recipe['iterations_per_epoch'] * args.epochs
            if recipe['epochs'] != args.epochs or recipe['max_iter'] != iterations or iterations != state['max_iter']:
                raise ValueError('Training duration does not match the complete DataLoader epoch')
            if args.eval_epochs is not None and recipe['checkpoint_save_iter'] != recipe['iterations_per_epoch']:
                raise ValueError('Epoch checkpoints were not configured for this training run')
        if completed['iteration'] != iterations or recipe['trajectories'] != args.expected_cases:
            raise ValueError('Unexpected training completion or initialization')
        verify_training_mode(recipe)
        if args.full_finetune:
            audit = batch.read_json(directory/'full_parameter_audit.json')
            if (audit['use_lora'] or audit['optimizer_covers_all_policy_parameters'] is not True
                    or audit['trainable_parameters'] != recipe['policy_parameters']):
                raise ValueError('Full-parameter optimizer audit failed')
        if recipe['trajectory_sources'] != expected_sources:
            raise ValueError('Training trajectory sources differ from the selected manifest')
        if (recipe['batch_size_per_gpu'] != args.batch_size_per_gpu
                or recipe['global_batch_size'] != args.batch_size_per_gpu*len(args.gpus)):
            raise ValueError('Training batch size differs from the requested recipe')
        metrics = [json.loads(line) for line in (directory/'train_metrics.jsonl').read_text().splitlines()]
        if len(metrics) != iterations or metrics[-1]['iteration'] != iterations:
            raise ValueError('Missing training steps')
        if any(not math.isfinite(row['sft_loss']) or row['use_mask_attention_loss'] for row in metrics):
            raise ValueError('Training metrics are invalid')
        return recipe
    def train(job_name, iterations, log_path):
        command = [str(Path(PYTHON).parent/'torchrun'), '--standalone',
                   '--nproc_per_node='+str(len(args.gpus)), str(train_script),
                   '--job-name', job_name, '--probe-every', '100',
                   '--manifest', str(args.manifest), '--data-root', str(args.data_root),
                   '--expected-cases', str(args.expected_cases),
                   '--batch-size-per-gpu', str(args.batch_size_per_gpu)]
        command.extend(['--full-finetune'] if args.full_finetune else
                       ['--lora-rank', str(args.lora_rank), '--lora-alpha', str(args.lora_alpha)])
        command.extend(['--epochs', str(args.epochs)] if iterations is None else ['--max-iter', str(iterations)])
        if args.eval_epochs is not None and iterations is None:
            command.append('--checkpoint-every-epoch')
        with log_path.open('w') as log:
            subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return verify_training(job_name, iterations)
    try:
        if args.continue_training_pid is not None:
            recipe = batch.read_json(run/'run.json')
            verify_training_mode(recipe)
            if (recipe['trajectories'] != args.expected_cases or recipe['max_iter'] != state['max_iter']
                    or recipe['epochs'] != args.epochs or recipe['use_mask_attention_loss']
                    or recipe['batch_size_per_gpu'] != args.batch_size_per_gpu):
                raise ValueError('Existing training recipe differs from the requested fresh one-epoch SFT')
            process = Path(f'/proc/{args.continue_training_pid}')
            update('TRAINING', existing_training_pid=args.continue_training_pid,
                   replaced_supervisor_pid=previous_state['pid'])
            while process.exists():
                try:
                    cmd = (process/'cmdline').read_bytes().replace(b'\0', b' ').decode()
                except FileNotFoundError:
                    break
                if not cmd.strip():
                    break
                if 'torchrun' not in cmd or str(train_script) not in cmd or args.job_name not in cmd:
                    raise ValueError('The existing training PID no longer belongs to this job')
                time.sleep(10)
            if not (run/'completed.json').exists():
                raise RuntimeError('Existing training process exited without completing the epoch')
            recipe = verify_training(args.job_name, args.max_iter)
        else:
            update('PREPARING', data_directory=str(args.data_root))
            with (run/'prepare.log').open('w') as log:
                subprocess.run([PYTHON, str(train_script), '--prepare-only', '--manifest', str(args.manifest),
                                '--data-root', str(args.data_root), '--expected-cases', str(args.expected_cases)], cwd=REPO, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            preflight = batch.read_json(args.data_root / 'preflight.json')
            expected_epoch_iterations = math.ceil(math.ceil(preflight['steps']/len(args.gpus))/args.batch_size_per_gpu)
            max_iter = args.max_iter if args.max_iter is not None else args.epochs * expected_epoch_iterations
            update('SMOKE_TRAINING', preflight=preflight, max_iter=max_iter,
                   iterations_per_epoch=expected_epoch_iterations,
                   smoke_iterations=2, smoke_log=str(run/'smoke_train.log'))
            smoke = train(args.job_name+'_smoke2', 2, run/'smoke_train.log')
            smoke_checkpoint = Path(smoke['checkpoint_directory'])/'iter_000000002'
            smoke_verification = verify_checkpoint(smoke_checkpoint)
            if args.full_finetune:
                smoke_export = REPO/'training'/f'{args.job_name}_smoke2_full.pt'
                update('SMOKE_EXPORTING', smoke_checkpoint_verification=smoke_verification)
                smoke_export_verification = export_checkpoint(smoke_checkpoint, smoke_export,
                                                              run/'smoke_export.log')
                update('SMOKE_EXPORT_VERIFIED', smoke_export=str(smoke_export),
                       smoke_export_verification=smoke_export_verification)
            update('TRAINING', smoke_checkpoint_verification=smoke_verification, train_log=str(run/'train.log'))
            recipe = train(args.job_name, args.max_iter, run/'train.log')
        multi_epoch = len(evaluation_epochs) > 1
        metrics = [json.loads(line) for line in (run/'train_metrics.jsonl').read_text().splitlines()]
        curve = []
        reports = {}
        for epoch in evaluation_epochs:
            iteration = recipe['max_iter'] if epoch is None else epoch * recipe['iterations_per_epoch']
            checkpoint_dir = Path(recipe['checkpoint_directory']) / f'iter_{iteration:09d}'
            suffix = f'_epoch{epoch:02d}' if multi_epoch else ''
            update('VERIFYING_CHECKPOINT', checkpoint_directory=str(checkpoint_dir),
                   current_evaluation_epoch=epoch, current_checkpoint_iteration=iteration)
            checkpoint_verification = verify_checkpoint(checkpoint_dir)
            extension = 'full' if args.full_finetune else 'merged'
            merged = REPO / 'training' / f'{args.job_name}_iter{iteration}_{extension}.pt'
            update('EXPORTING' if args.full_finetune else 'MERGING',
                   checkpoint_verification=checkpoint_verification, merged_checkpoint=str(merged))
            merge_verification = export_checkpoint(checkpoint_dir, merged, run/f'merge{suffix}.log')
            evaluation = REPO / 'experiments/libero_plus/sft190_p95' / args.job_name
            if multi_epoch:
                evaluation = evaluation / f'epoch_{epoch:02d}'
            command = [PYTHON, str(REPO/'scripts/run_libero_plus_batch.py'), '--ckpt',str(merged),
                       '--output-root',str(evaluation),'--gpus',*map(str,args.gpus),
                       '--action-seed','42','--environment-seed','10000','--settle-steps','30',
                       '--max-steps','700','--save-episode-data', '--task-ids-file', str(selection_path)]
            eval_env = dict(env)
            for key in list(eval_env):
                if key.startswith(('COSMOS_TTA_', 'TTA_REPAIR_')):
                    eval_env.pop(key)
            update('VALIDATING_EVALUATION', merge_verification=merge_verification,
                   evaluation_directory=str(evaluation), eval_log=str(run/f'eval{suffix}.log'))
            with (run/f'eval_preflight{suffix}.log').open('w') as log:
                subprocess.run(command+['--validate-only'],cwd=REPO,env=eval_env,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            update('EVALUATING')
            with (run/f'eval{suffix}.log').open('w') as log:
                result = subprocess.run(command,cwd=REPO,env=eval_env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode != 0:
                summary_path = evaluation/'summary.json'
                summary = batch.read_json(summary_path) if summary_path.is_file() else {}
                if not summary.get('errors'):
                    raise RuntimeError(f'Evaluation exited with code {result.returncode}')
                update('RETRYING_EVALUATION_ERRORS', errors=summary['errors'])
                with (run/f'eval_retry_errors{suffix}.log').open('w') as log:
                    subprocess.run(command+['--resume','--retry-errors'],cwd=REPO,env=eval_env,
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
            report = compare_results(evaluation, selected, merged, args.evaluation_scope)
            report.update(training_mode=recipe.get('training_mode', 'lora'),
                          use_lora=not args.full_finetune,
                          trainable_parameters=recipe.get('trainable_parameters'))
            window = metrics[max(0,iteration-100):iteration]
            report.update(epoch=epoch, iteration=iteration,
                          training_loss_last_100=sum(row['sft_loss'] for row in window)/len(window),
                          checkpoint_learning_rate=window[-1]['learning_rate'])
            report_path = run/'epoch_reports'/f'epoch_{epoch:02d}.json' if epoch is not None else run/'epoch_reports/final.json'
            batch.atomic_json(report_path, report)
            reports[str(epoch)] = report
            row = dict(epoch=epoch, iteration=iteration, training_loss_last_100=report['training_loss_last_100'])
            for label, category in [('training','training_cases'),('heldout','heldout_baseline_failures')]:
                partition = report['by_partition'].get(category, {})
                row.update({label+'_successes':partition.get('successes',0),
                            label+'_total':partition.get('total',0),
                            label+'_success_rate':partition.get('success_rate')})
            row.update(overall_successes=report['overall']['successes'],
                       overall_total=report['overall']['total'],
                       overall_success_rate=report['overall']['success_rate'])
            curve.append(row)
            batch.atomic_json(run/'evaluation_curve.json', dict(epochs=curve, protocol=report['protocol']))
            table = io.StringIO()
            writer = csv.DictWriter(table, fieldnames=list(curve[0]))
            writer.writeheader();writer.writerows(curve)
            temporary = run/'evaluation_curve.csv.tmp'
            temporary.write_text(table.getvalue());temporary.replace(run/'evaluation_curve.csv')
            update('EPOCH_EVALUATED', evaluated_epochs=[row['epoch'] for row in curve],
                   epoch_results=curve, evaluation_curve=str(run/'evaluation_curve.json'))
        report = dict(report, epoch_reports=reports, epoch_curve=curve)
        batch.atomic_json(run/'evaluation_report.json', report)
        update('COMPLETED', evaluation_report=str(run/'evaluation_report.json'),
               overall=report['overall'], by_partition=report['by_partition'], by_category=report['by_category'])
    except Exception as error:
        update('FAILED', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
