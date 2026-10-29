'''Paired soft-source tests on a fixed checkpoint, without optimizer updates.'''

import json

import numpy as np
import torch

from diagnose_scale_propagation import (
    forward, point_groups, error_metrics, stage_report, original)
from diagnose_hierarchical_rejection import pool_metrics, paired_comparison


def policies(strengths):
    result = {'baseline': None}
    for strength in strengths:
        for scope, levels in [('r0', [0]), ('all', list(range(5)))]:
            for mode, good_only in [('good_only', True), ('accepted', False)]:
                name = '{}_{}_lambda{:g}'.format(scope, mode, strength)
                result[name] = dict(kind='soft_anchor', levels=levels,
                                    good_only=good_only, strength=strength)
    return result


def anchor_errors(probe, labels):
    '''Compare native-scale metric depth to its selected sparse measurement.

    At coarse levels this measures anchor fidelity, not coarse-cell GT accuracy:
    the selected measurement may lie on only one of the surfaces in that cell.
    '''
    rows = []
    for level, stage, depth in probe.stages:
        if stage not in ('before_propagation', 'after_propagation', 'after_refinement'):
            continue
        context = probe.contexts[level]
        kept = torch.from_numpy(context['keep']) > 0
        sparse = torch.from_numpy(context['sparse'])
        rows.append(dict(level=level, stage=stage, groups={
            name: error_metrics(torch.from_numpy(depth), sparse, kept & group)
            for name, group in [('good', labels[level] == 0),
                                ('bad', labels[level] == 1),
                                ('unknown', labels[level] == -1)]}))
    return rows


def local_replay(base, labels, args, sample, seed, emit):
    '''Only the update equation changes: geometry, weights and ratios are fixed.

    Replaying from the same initial input for t iterations uses the production
    function and avoids maintaining a separate approximate propagation loop.
    '''
    for level, context in base.contexts.items():
        values = {key: torch.from_numpy(value) for key, value in context.items()}
        iterations = args.n_propagation_iterations[4 - level]
        kept = values['keep'] > 0
        for strength in [0.0] + list(dict.fromkeys(args.anchor_strengths)):
            for mode in (['accepted'] if strength == 0 else ['good_only', 'accepted']):
                mask = (labels[level] == 0).to(values['keep']) if mode == 'good_only' else None
                trajectory = []
                for step in range(iterations + 1):
                    scale = original.propagate_scale(
                        values['scale'], values['ratio'], values['keep'], values['logits'],
                        step, strength, mask)
                    depth = scale * values['geometry']
                    trajectory.append(dict(iteration=step, groups={
                        name: error_metrics(depth, values['sparse'], kept & group)
                        for name, group in [('good', labels[level] == 0),
                                            ('bad', labels[level] == 1)]}))
                emit('anchor_local_replay', sample=sample, seed=seed, level=level,
                     strength=strength, mode=mode, trajectory=trajectory)


def diagnose_anchor_sample(wrapper, image, sparse, target, args, sample,
                           directory, emit, save_maps=False):
    target = target.cpu()
    valid = torch.isfinite(target) & (target > max(0, args.min_evaluate_depth)) & \
        (target < args.max_evaluate_depth)
    device_target = target.to(wrapper.device)
    trials = policies(args.anchor_strengths)
    for seed_index, seed in enumerate(args.seeds):
        actual_seed = (seed + 1000003 * sample) % (2 ** 32)
        prediction, base = forward(wrapper, image, sparse, device_target, args, actual_seed)
        frozen = base.frozen
        replay, replay_probe = forward(
            wrapper, image, sparse, device_target, args, actual_seed, frozen)
        difference = float((prediction - replay).abs().max())
        emit('anchor_replay', sample=sample, seed=seed, max_depth_difference_m=difference)
        if not torch.isfinite(prediction).all() or difference > 1e-5:
            raise RuntimeError('Nonfinite baseline or failed frozen-mask replay.')
        del replay_probe, replay
        labels = {r: point_groups(sparse.cpu(), target, frozen[r].shape[-2:])[2]
                  for r in range(5)}
        indices = valid.flatten().nonzero().flatten()
        if not indices.numel():
            raise ValueError('No evaluated GT for sample {}'.format(sample))
        squared = (prediction - target).square()
        worst = torch.zeros_like(valid)
        worst.flatten()[indices[torch.topk(squared.flatten()[indices],
                                          max(1, int(indices.numel() * .01))).indices]] = True
        groups = {'baseline_worst_1pct': worst,
                  'accepted_good_r0': (frozen[0].cpu() > 0) & (labels[0] == 0),
                  'accepted_bad_r0': (frozen[0].cpu() > 0) & (labels[0] == 1),
                  'away_sparse': sparse.cpu() <= 0,
                  'far_gt_over_40m': target > 40}
        maps = {'target': target.numpy(), 'sparse': sparse.cpu().numpy(),
                'evaluated': valid.numpy(), 'baseline_worst_1pct': worst.numpy()}
        # Per-iteration native trajectories are collected on the saved subset.
        if save_maps and seed_index == 0:
            local_replay(base, labels, args, sample, seed, emit)
        baseline_sse = float(squared[valid].double().sum())
        baseline_objective = base.losses['objective']
        for name, intervention in trials.items():
            if intervention is None:
                output, probe = prediction, base
            else:
                output, probe = forward(wrapper, image, sparse, device_target, args,
                                        actual_seed, frozen, intervention)
            if any(not torch.equal(probe.frozen[r], frozen[r]) for r in range(5)):
                raise RuntimeError('Rejection decisions changed in a paired anchor test.')
            metrics = error_metrics(output, target, valid)
            emit('anchor_result', sample=sample, seed=seed, actual_seed=actual_seed,
                 policy=name, intervention=intervention, metrics=metrics,
                 delta_sse_vs_baseline=metrics.get('sse_m2', float('nan')) - baseline_sse,
                 delta_objective_vs_baseline=probe.losses['objective'] - baseline_objective,
                 losses=probe.losses,
                 groups={key: error_metrics(output, target, valid & mask)
                         for key, mask in groups.items()},
                 stages=stage_report(probe, target, valid, worst),
                 anchor_fidelity=anchor_errors(probe, labels))
            if save_maps and seed_index == 0:
                maps[name + '_prediction'] = output.numpy()
                for level, stage, depth in probe.stages:
                    maps['{}_r{}_{}'.format(name, level, stage)] = depth
            if intervention is not None:
                del probe
        if save_maps and seed_index == 0:
            for r, context in base.contexts.items():
                for key, value in context.items():
                    maps['baseline_r{}_{}'.format(r, key)] = value
                maps['r{}_labels'.format(r)] = labels[r].numpy()
            np.savez_compressed(str(directory / 'anchor_sample_{}_seed_{}.npz'.format(sample, seed)), **maps)
        print('anchor sample {}, seed {} complete'.format(sample, seed), flush=True)


def write_anchor_summary(directory, records):
    rows = [row for row in records if row['type'] == 'anchor_result']
    names = list(dict.fromkeys(row['policy'] for row in rows))
    summary = {'policies': {}, 'comparisons': []}
    for name in names:
        selected = [row for row in rows if row['policy'] == name]
        summary['policies'][name] = dict(
            metrics=pool_metrics([r['metrics'] for r in selected]),
            groups={key: pool_metrics([r['groups'][key] for r in selected])
                    for key in selected[0]['groups']},
            mean_objective=float(np.mean([r['losses']['objective'] for r in selected])))
        if name != 'baseline':
            summary['comparisons'].append(paired_comparison(rows, name, 'baseline'))
    (directory / 'anchor_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = ['# Persistent soft-anchor diagnosis', '',
             'Frozen checkpoint and identical rejection decisions. No training or hard restoration.',
             'baseline: old update (lambda=0). r0: modify only full resolution. all: modify every level.',
             'good_only: oracle tether only at accepted GT-verified good measurements;',
             'initial injection is unchanged, including bad/unknown accepted measurements.',
             'accepted: deployable tether at every accepted measurement, without GT.', '',
             '| Policy | MAE mm | RMSE mm |', '|---|---:|---:|']
    for name, result in summary['policies'].items():
        metric = result['metrics']
        lines.append('| {} | {} | {} |'.format(name,
            metric.get('mae_m', float('nan')) * 1000 if metric.get('mae_m') is not None else 'nonfinite',
            metric.get('rmse_m', float('nan')) * 1000 if metric.get('rmse_m') is not None else 'nonfinite'))
    lines += ['', '## What to inspect', '',
              '- JSONL anchor_local_replay: identical block inputs; t=0 is immediately after injection.',
              '- anchor_fidelity: good/bad-anchor errors before propagation, after propagation, after refinement.',
              '- stages: common baseline worst-1% error set through all levels; no per-policy reselection.',
              '- groups: accepted good/bad anchors, missing sparse pixels, and far-depth evaluated pixels.',
              '- Summary comparisons bootstrap images after seed averaging, not pixels or seeds independently.',
              '- If local fidelity improves but final RMSE does not, inspect refinement and downstream changes.',
              '- good_only gains alone do not prove a deployable improvement; accepted can reinforce outliers.',
              '- Coarse anchor fidelity is to the selected measurement, not a claim of correct entire-cell depth.',
              '- These are checkpoint interventions, not evidence of converged retrained performance.',
              '- Lambda is a diagnostic sweep; confirm selection on a separate validation split.',
              '- Share the generated ZIP, including provenance, source snapshots, JSONL, summary, and maps.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
