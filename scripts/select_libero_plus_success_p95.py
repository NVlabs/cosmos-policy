#!/usr/bin/env python3
"""Select complete successful Memory repairs using same-base-task demo P95.

Lengths include the baseline prefix, Memory intervention and continuation, but
exclude settling actions. Thresholds come from P95_RECORD (a committed
summary.json) when it exists, otherwise from the demos in --demo-dir.
Uses only saved metadata, NumPy and h5py; no GPU/model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import h5py
import numpy as np

from collect_libero_plus_failures import (
    REPO, PROTOCOL_FIELDS, catalog_from_config, digest, json_payload,
    protocol_from_config, read_json, recorded_file, validate_identity,
    validate_protocol, write_payloads,
)

RULE = 'completed and success == true and full action_steps <= P95 of regenerated success demonstrations for the same base task'
LENGTH_DEFINITION = 'full saved actions; baseline prefix + Memory intervention + policy continuation; excludes 30 settling actions'
EVALUATION_SCOPES = {
    'robot_initial_states': 'robotinit-failures', 'camera_viewpoints': 'camera-failures',
    'sensor_noise': 'sensor-noise-failures', 'background_textures': 'background-failures',
    'light_conditions': 'light-failures',
}


P95_RECORD = REPO / 'experiments/tta/repair_libero_plus_435/success_task_p95/summary.json'
THRESHOLD_FIELDS = ('training_file', 'training_episodes', 'p95', 'max_integer_steps')


def recorded_thresholds(record_path, base_tasks):
    """Per-base-task demo P95 thresholds from a committed selection record.

    The record is written by an earlier run of this script. Only the fields
    demo_threshold() returns are kept, so a selection driven by the record is
    byte-identical to one driven by the demonstration files.
    """
    recorded = read_json(record_path)['by_task']
    missing = [base for base in base_tasks if base not in recorded]
    if missing:
        raise ValueError(f'P95 record {record_path} lacks base tasks: {missing}')
    return {base: {key: recorded[base][key] for key in THRESHOLD_FIELDS} for base in base_tasks}


def demo_threshold(demo_dir, base_task):
    if not isinstance(base_task, str) or not base_task or '/' in base_task or base_task in ('.', '..'):
        raise ValueError(f'Invalid base task: {base_task!r}')
    path = Path(demo_dir) / (base_task + '_demo.hdf5')
    with h5py.File(path, 'r') as data:
        lengths = []
        for key in sorted(data['data']):
            shape = data['data'][key]['actions'].shape
            if len(shape) != 2 or shape[1] != 7 or shape[0] <= 0:
                raise ValueError(f'Invalid demonstration actions: {path}:{key}')
            lengths.append(shape[0])
    if not lengths:
        raise ValueError(f'No success demonstrations: {path}')
    threshold = float(np.percentile(lengths, 95, method='linear'))
    return dict(training_file=str(path), training_episodes=len(lengths),
                p95=threshold, max_integer_steps=math.floor(threshold))


def validate_success_trajectory(result, result_path):
    artifacts = {key: recorded_file(result.get(key), result_path.parent)
                 for key in ('video', 'actions', 'episode_hdf5')}
    steps = result.get('action_steps')
    if type(steps) is not int or steps <= 0:
        raise ValueError(f'Invalid action_steps: {result_path}')
    with h5py.File(artifacts['episode_hdf5'], 'r') as data:
        if data.attrs.get('success') != True:
            raise ValueError(f'Success attribute disagrees with repair result: {result_path}')
        if data['actions'].shape != (steps, 7) or data['proprio'].shape != (steps, 9):
            raise ValueError(f'Trajectory shape/length mismatch: {result_path}')
        for key in ('primary_images_jpeg', 'wrist_images_jpeg'):
            if len(data[key]) != steps:
                raise ValueError(f'Image/action length mismatch: {result_path}')
        actions = data['actions'][:]
        if not np.isfinite(actions).all() or not np.isfinite(data['proprio'][:]).all():
            raise ValueError(f'Non-finite trajectory: {result_path}')
        saved_actions = np.load(artifacts['actions'], allow_pickle=False)
        if saved_actions.shape != actions.shape or not np.array_equal(saved_actions, actions):
            raise ValueError(f'actions.npy disagrees with HDF5: {result_path}')
    if type(result.get('t_star')) is not int or not 0 <= result['t_star'] < steps:
        raise ValueError(f'Invalid t_star: {result_path}')
    return {key: str(path) for key, path in artifacts.items()}


def build_selection(failure_manifest, repair_runs, demo_dir, allow_incomplete=False):
    failure_manifest, demo_dir = Path(failure_manifest).resolve(), Path(demo_dir).resolve()
    failures = read_json(failure_manifest)
    baseline = Path(failures['source_run'])
    config_path = baseline / 'run_config.json'
    config = read_json(config_path)
    if digest(config_path) != failures['source_config_sha256']:
        raise ValueError('Baseline run_config changed since failure-manifest generation')
    protocol = protocol_from_config(config)
    if protocol != failures['protocol']:
        raise ValueError('Failure manifest protocol differs from baseline')
    catalog = catalog_from_config(config)
    cases = {}
    for row in failures['cases']:
        task_id = row['task_id']
        if task_id not in catalog or task_id in cases:
            raise ValueError(f'Unknown/duplicate failure task_id: {task_id}')
        validate_identity(row, catalog[task_id], failure_manifest)
        validate_protocol(row, protocol, failure_manifest)
        if row.get('status') != 'completed' or row.get('success') is not False:
            raise ValueError(f'Failure manifest includes a nonfailure: {task_id}')
        if row['case_id'] != f'task{task_id:04d}' or not row['task_name'].startswith(row['base_task'] + '_'):
            raise ValueError(f'Invalid failure case/base task: {task_id}')
        cases[task_id] = row
    if failures['num_cases'] != len(cases):
        raise ValueError('Failure manifest count mismatch')
    roots = [Path(root).resolve() for root in repair_runs]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError('Repair run directories must be nonempty and distinct')
    results = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        run_config = root / 'run_config.json'
        if run_config.is_file():
            metadata = read_json(run_config)
            if metadata.get('source_config_sha256', failures['source_config_sha256']) != failures['source_config_sha256']:
                raise ValueError(f'Repair run uses a different baseline: {root}')
        for path in sorted(root.glob('*/task*/result.json')):
            result = read_json(path)
            task_id = result['task_id']
            if task_id not in cases or task_id in results:
                raise ValueError(f'Unknown/duplicate repaired task_id {task_id}: {path}')
            source = cases[task_id]
            validate_identity(result, source, path)
            if path.parent.name != source['case_id'] or path.parent.parent.name != source['group']:
                raise ValueError(f'Repair case directory mismatch: {path}')
            validate_protocol(result.get('protocol', {}), protocol, path, PROTOCOL_FIELDS[:5])
            for key in PROTOCOL_FIELDS[5:]:
                if key in result.get('protocol', {}) and result['protocol'][key] != protocol[key]:
                    raise ValueError(f'Repair protocol mismatch ({key}): {path}')
            if result.get('baseline_success') is not False:
                raise ValueError(f'Repair is not from a baseline failure: {path}')
            if result.get('status') == 'completed' and type(result.get('success')) is not bool:
                raise ValueError(f'Completed repair requires a Boolean success: {path}')
            results[task_id] = (path, result)
    incomplete = [{'task_id': task_id, 'status': results[task_id][1].get('status') if task_id in results else 'missing'}
                  for task_id in sorted(cases)
                  if task_id not in results or results[task_id][1].get('status') != 'completed']
    if (incomplete or failures.get('incomplete')) and not allow_incomplete:
        raise ValueError(f'Correction inputs incomplete: {len(incomplete)} unfinished repairs; '
                         'finish the run or explicitly use --allow-incomplete')
    bases = sorted({row['base_task'] for row in cases.values()})
    thresholds = (recorded_thresholds(P95_RECORD, bases) if P95_RECORD.is_file()
                  else {base: demo_threshold(demo_dir, base) for base in bases})
    selected, excluded, memory_failed = [], [], []
    for task_id, source in sorted(cases.items()):
        if task_id not in results or results[task_id][1].get('status') != 'completed':
            continue
        path, result = results[task_id]
        if result['success'] is False:
            memory_failed.append(source)
            continue
        artifacts = validate_success_trajectory(result, path)
        threshold = thresholds[source['base_task']]
        row = dict(task_id=task_id, case_id=source['case_id'], base_task=source['base_task'],
                   group=source['group'], task_name=source['task_name'], success=True,
                   action_steps=result['action_steps'], training_p95=threshold['p95'],
                   max_integer_steps=threshold['max_integer_steps'], t_star=result['t_star'],
                   video=artifacts['video'], actions=artifacts['actions'],
                   episode_hdf5=artifacts['episode_hdf5'], repair_result=str(path))
        (selected if row['action_steps'] <= row['training_p95'] else excluded).append(row)
    length_definition = LENGTH_DEFINITION.replace('30 settling actions', f"{protocol['settle_steps']} settling actions")
    meta = dict(version=1, rule=RULE, percentile_method='numpy linear',
                length_definition=length_definition, training_data=str(demo_dir), protocol=protocol)
    if incomplete or failures.get('incomplete'):
        meta['incomplete'] = True
    selected_manifest = dict(meta, num_cases=len(selected), cases=selected)
    excluded_manifest = dict(meta, num_cases=len(excluded), reason='full successful trajectory exceeds same-task training P95', cases=excluded)
    all_success = selected + excluded
    by_task = {}
    for base, threshold in thresholds.items():
        before = sum(row['base_task'] == base for row in all_success)
        kept = sum(row['base_task'] == base for row in selected)
        by_task[base] = dict(successful_before_filter=before, kept=kept, excluded=before-kept, **threshold)
    partitions, by_category = {}, {}
    trained_ids = {row['task_id'] for row in selected}
    for group in sorted({row['group'] for row in cases.values()}):
        training = [row for row in selected if row['group'] == group]
        heldout = [row for task_id, row in sorted(cases.items()) if row['group'] == group and task_id not in trained_ids]
        failed = [row for row in memory_failed if row['group'] == group]
        before = sum(row['group'] == group for row in all_success)
        by_category[group] = dict(successful_before_filter=before, kept=len(training), excluded=before-len(training))
        partitions[group] = dict(training=training, heldout=heldout, memory_failed=failed,
                                 evaluation=[dict(task_id=task_id, task_name=row['task_name'], group=group,
                                                  baseline_success=False, in_training=task_id in trained_ids)
                                             for task_id, row in sorted(cases.items()) if row['group'] == group])
    summary = dict(rule=RULE, percentile_method='numpy linear', length_definition=length_definition,
                   training_data=str(demo_dir), successful_before_filter=len(all_success), kept=len(selected),
                   excluded=len(excluded), kept_fraction=len(selected)/len(all_success) if all_success else 0.0,
                   mean_kept_steps=float(np.mean([row['action_steps'] for row in selected])) if selected else None,
                   median_kept_steps=float(np.median([row['action_steps'] for row in selected])) if selected else None,
                   by_task=by_task, by_category=by_category)
    generation = dict(failure_manifest=str(failure_manifest), failure_manifest_sha256=digest(failure_manifest),
                      source_config_sha256=failures['source_config_sha256'], repair_runs=[str(root) for root in roots],
                      baseline_failures=len(cases), completed_repairs=len(cases)-len(incomplete),
                      memory_successes=len(all_success), memory_failures=len(memory_failed),
                      selected=len(selected), p95_excluded=len(excluded), complete=not incomplete and not failures.get('incomplete', False),
                      incomplete_repairs=incomplete, demo_thresholds=thresholds)
    return dict(selected_manifest=selected_manifest, excluded_manifest=excluded_manifest,
                summary=summary, generation=generation, partitions=partitions)


def selection_payloads(selection, output_dir, overwrite=False):
    output_dir = Path(output_dir).resolve()
    payloads = {}
    for name in ('selected_manifest', 'excluded_manifest', 'summary', 'generation'):
        path = output_dir / (name + '.json')
        payloads[path] = json_payload(path, selection[name], overwrite)
    selected_path = output_dir / 'selected_manifest.json'
    selection_hash = hashlib.sha256(payloads[selected_path]).hexdigest()
    meta = {key: value for key, value in selection['selected_manifest'].items() if key not in ('cases', 'num_cases')}
    for group, partition in selection['partitions'].items():
        root = output_dir / 'by_category' / group
        for key, name in [('training', 'training_manifest'), ('heldout', 'heldout_manifest'), ('memory_failed', 'memory_failed_manifest')]:
            value = dict(meta, selection_scope=group, partition=key,
                         source_selection_manifest=str(selected_path), source_selection_manifest_sha256=selection_hash,
                         num_cases=len(partition[key]), cases=partition[key])
            if key == 'training':
                value['num_action_steps'] = sum(row['action_steps'] for row in partition[key])
                if group in EVALUATION_SCOPES:
                    value['evaluation_subset'] = dict(
                        scope=EVALUATION_SCOPES[group], num_cases=len(partition['evaluation']),
                        task_ids=[row['task_id'] for row in partition['evaluation']],
                    )
            else:
                value['rule'] = ('baseline failures outside the selected training cases' if key == 'heldout'
                                 else 'completed Memory repairs with success == false')
            path = root / (name + '.json')
            payloads[path] = json_payload(path, value, overwrite)
        evaluation = dict(evaluation_scope=EVALUATION_SCOPES.get(group, 'baseline-failures'), task_ids=[row['task_id'] for row in partition['evaluation']], cases=partition['evaluation'])
        path = root / 'evaluation_cases.json'
        payloads[path] = json_payload(path, evaluation, overwrite)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['task', 'training_episodes', 'training_p95', 'max_integer_steps', 'successful_before_filter', 'kept', 'excluded'])
    for base, row in sorted(selection['summary']['by_task'].items()):
        writer.writerow([base, row['training_episodes'], row['p95'], row['max_integer_steps'], row['successful_before_filter'], row['kept'], row['excluded']])
    payloads[output_dir / 'task_summary.csv'] = stream.getvalue().encode('utf-8')
    return payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repair_root = REPO / 'experiments/tta/repair_libero_plus_435'
    parser.add_argument('--failure-manifest', type=Path, default=repair_root / 'inputs/failed_435.json')
    parser.add_argument('--repair-runs', type=Path, nargs='+', default=[repair_root / 'smoke_4', repair_root / 'remaining_431'])
    parser.add_argument('--demo-dir', type=Path, default=REPO / 'LIBERO-Cosmos-Policy/success_only/libero_10_regen')
    parser.add_argument('--output-dir', type=Path, default=repair_root / 'success_task_p95')
    parser.add_argument('--allow-incomplete', action='store_true', help='Explicitly omit unfinished repairs; do not count them as Memory failures')
    parser.add_argument('--overwrite', action='store_true', help='Replace different existing outputs')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and print counts without writing files')
    args = parser.parse_args()
    selection = build_selection(args.failure_manifest, args.repair_runs, args.demo_dir, args.allow_incomplete)
    if not args.dry_run:
        write_payloads(selection_payloads(selection, args.output_dir, args.overwrite), args.overwrite)
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()), dry_run=args.dry_run,
                         **{key: selection['generation'][key] for key in ('baseline_failures', 'memory_successes', 'memory_failures', 'selected', 'p95_excluded', 'complete')},
                         by_category=selection['summary']['by_category']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
