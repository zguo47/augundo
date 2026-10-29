'''Controlled A--E rejection policies, without training or production changes.

GT-guided decisions test whether measurement-consistency rejection can help.
They are neither a deployable method nor an upper bound over optimal masks.
'''

from collections import defaultdict
import json

import numpy as np
import torch

from diagnose_scale_propagation import (
    forward, point_groups, error_metrics, boundary_mask, stage_report)


POLICIES = ['A_keep_all', 'B_r0_only', 'C_learned'] + [
    'D_oracle_r{}'.format(r) for r in range(1, 5)] + ['E_oracle_hierarchy']


def policy_masks(learned, selected, labels):
    '''Keep the same baseline r0 decisions in B--E, including oracle policies.

    Each oracle is a fresh complete mask: keep known good AND ambiguous/unknown
    measurements, reject known bad, and never activate missing measurements.
    Thus it repairs false rejections as well as false acceptances. Labels come
    from GT at the selected measurement's original coordinate, not pooled GT.
    '''
    available = {r: (selected[r] > 0).to(learned[r]) for r in range(5)}
    oracle = {r: available[r] * (labels[r] != 1).to(learned[r]) for r in range(5)}
    full_res_only = dict(available)
    full_res_only[0] = learned[0]
    masks = {'A_keep_all': available, 'B_r0_only': full_res_only, 'C_learned': learned}
    for r in range(1, 5):
        masks['D_oracle_r{}'.format(r)] = dict(full_res_only)
        masks['D_oracle_r{}'.format(r)][r] = oracle[r]
    masks['E_oracle_hierarchy'] = {r: oracle[r] if r else learned[0] for r in range(5)}
    return masks


def rejection_counts(masks, selected, labels, learned):
    result = {}
    for r in range(5):
        mask, label = masks[r].cpu(), labels[r]
        available = selected[r] > 0
        result['r{}'.format(r)] = dict(
            available=int(available.sum()), kept=int(mask.sum()),
            good_kept=int(((label == 0) & (mask > 0)).sum()),
            good_rejected=int(((label == 0) & (mask == 0)).sum()),
            bad_kept=int(((label == 1) & (mask > 0)).sum()),
            bad_rejected=int(((label == 1) & (mask == 0)).sum()),
            unknown_or_ambiguous=int(((label == -1) & available).sum()),
            known_good_restored_vs_learned=int(((label == 0) & (mask > 0) & (learned[r].cpu() == 0)).sum()))
    return result


def diagnose_hierarchy_sample(wrapper, image, sparse, target, args, sample,
                              directory, emit, save_maps=False):
    '''Repeat paired policies with shared sampled decisions for each seed.'''
    device_target = target.to(wrapper.device)
    target = target.cpu()
    evaluated = torch.isfinite(target) & (target > max(0.0, args.min_evaluate_depth)) & \
        (target < args.max_evaluate_depth)
    if not evaluated.any():
        raise ValueError('No evaluated GT for sample {}'.format(sample))
    groups = {'at_sparse': sparse.cpu() > 0, 'away_sparse': sparse.cpu() <= 0,
              'gt_boundary_5px': boundary_mask(target)}
    for low, high in [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, float('inf'))]:
        groups['depth_{}_{}'.format(low, high)] = (target >= low) & (target < high)
    emit('hierarchy_input', sample=sample, shape=list(image.shape),
         sparse_vs_gt=error_metrics(sparse.cpu(), target, evaluated & (sparse.cpu() > 0)))
    for seed_index, seed in enumerate(args.seeds):
        # Separate images' random streams even when adjacent seeds are used.
        actual_seed = (seed + 1000003 * sample) % (2 ** 32)
        current, baseline = forward(wrapper, image, sparse, device_target, args, actual_seed)
        learned = baseline.frozen
        selected, labels, indices = {}, {}, {}
        for r in range(5):
            selected[r], indices[r], labels[r], _ = point_groups(
                sparse.cpu(), target, learned[r].shape[-2:])
        masks = policy_masks(learned, selected, labels)
        replay, replay_probe = forward(wrapper, image, sparse, device_target, args, actual_seed, learned)
        replay_delta = float((current - replay).abs().max())
        objective_delta = replay_probe.losses['objective'] - baseline.losses['objective']
        emit('hierarchy_replay', sample=sample, seed=seed, actual_seed=actual_seed,
             max_depth_difference_m=replay_delta, objective_difference=objective_delta)
        if not torch.isfinite(current).all() or replay_delta > 1e-5 or abs(objective_delta) > 1e-6:
            raise RuntimeError('Nonfinite baseline or failed frozen-mask replay.')
        del replay_probe, replay
        # The same baseline worst-pixel set is used in EVERY policy this seed.
        valid_indices = evaluated.flatten().nonzero().flatten()
        squared = (current - target).square()
        worst = torch.zeros_like(evaluated)
        count = max(1, int(valid_indices.numel() * 0.01))
        worst.flatten()[valid_indices[torch.topk(squared.flatten()[valid_indices], count).indices]] = True
        groups['baseline_worst_1pct'] = worst

        def measure(prediction, probe):
            return dict(prediction=prediction, losses=probe.losses,
                        metrics=error_metrics(prediction, target, evaluated),
                        stages=stage_report(probe, target, evaluated, worst),
                        groups={name: error_metrics(prediction, target, evaluated & group)
                                for name, group in groups.items()})

        results = {'C_learned': measure(current, baseline)}
        del baseline
        for policy in [name for name in POLICIES if name != 'C_learned']:
            prediction, probe = forward(wrapper, image, sparse, device_target, args,
                                        actual_seed, masks[policy])
            # Ensure the override actually controlled every gate, especially r0.
            if any(not torch.equal(probe.frozen[r], masks[policy][r]) for r in range(5)):
                raise RuntimeError('A rejection policy was not applied exactly.')
            results[policy] = measure(prediction, probe)
            del probe
        reference = results['B_r0_only']
        reference_error = (reference['prediction'] - target).square()
        for policy in POLICIES:
            value = results[policy]
            difference = ((value['prediction'] - target).square() - reference_error)[evaluated]
            stage_deltas = [dict(level=stage['level'], stage=stage['stage'],
                                 delta_sse_m2=stage['sse_m2'] - prior['sse_m2'])
                            for stage, prior in zip(value['stages'], reference['stages'])]
            emit('hierarchy_result', sample=sample, seed=seed, actual_seed=actual_seed,
                 policy=policy, metrics=value['metrics'], groups=value['groups'], losses=value['losses'],
                 stages=value['stages'], stage_deltas_vs_B=stage_deltas,
                 rejection=rejection_counts(masks[policy], selected, labels, learned),
                 delta_objective_vs_B=value['losses']['objective'] - reference['losses']['objective'],
                 delta_sse_vs_B=float(difference.double().sum()),
                 sse_added_vs_B=float(difference.clamp_min(0).double().sum()),
                 sse_removed_vs_B=float((-difference).clamp_min(0).double().sum()),
                 improved_pixels_vs_B=int((difference < 0).sum()),
                 worsened_pixels_vs_B=int((difference > 0).sum()))
        if save_maps and seed_index == 0:
            maps = {'evaluated_mask': evaluated.numpy(), 'baseline_worst_mask': worst.numpy()}
            for policy, value in results.items():
                maps[policy + '_prediction'] = value['prediction'].numpy()
                maps[policy + '_squared_error_delta_vs_B'] = (
                    ((value['prediction'] - target).square() - reference_error) * evaluated).numpy()
            for r in range(5):
                maps['r{}_labels'.format(r)] = labels[r].numpy()
                maps['r{}_original_indices'.format(r)] = indices[r].numpy()
                for policy in POLICIES:
                    maps['{}_r{}_keep'.format(policy, r)] = masks[policy][r].cpu().numpy()
            np.savez_compressed(str(directory / 'hierarchy_sample_{}_seed_{}.npz'.format(sample, seed)), **maps)
        print('hierarchy sample {}, seed {}: A--E complete'.format(sample, seed), flush=True)


def pool_metrics(metrics):
    '''Pixel-weighted, averaging squared error over images AND sampled seeds.'''
    metrics = [m for m in metrics if m['count']]
    n = sum(m['count'] for m in metrics)
    result = {'count_including_seed_repeats': n}
    if not n:
        return result
    for name in ('mae_m', 'rmse_m', 'imae_per_km', 'irmse_per_km'):
        if any(name not in m or not np.isfinite(m[name]) for m in metrics):
            result[name] = None
            continue
        power = 2 if name in ('rmse_m', 'irmse_per_km') else 1
        result[name] = (sum(m[name] ** power * m['count'] for m in metrics) / n) ** (1.0 / power)
    return result


def paired_comparison(rows, policy, reference):
    '''Bootstrap images, not pixels or stochastic repeats as independent data.

    CI is for the mean paired per-image RMSE difference after seed averaging;
    it is NOT the confidence interval for pooled RMSE in the main table.
    '''
    table = {(r['sample'], r['seed'], r['policy']): r for r in rows}
    per_image = defaultdict(list)
    for row in rows:
        if row['policy'] != policy:
            continue
        other = table.get((row['sample'], row['seed'], reference))
        if other is None:
            continue
        if 'rmse_m' not in row['metrics'] or 'rmse_m' not in other['metrics']:
            continue
        per_image[row['sample']].append(row['metrics']['rmse_m'] - other['metrics']['rmse_m'])
    values = np.array([np.mean(v) for v in per_image.values()])
    result = dict(policy=policy, reference=reference, n_images=len(values))
    if len(values):
        result.update(mean_image_rmse_delta_m=float(values.mean()),
                      images_improved=int((values < 0).sum()), images_worsened=int((values > 0).sum()))
    if len(values) >= 2:
        random = np.random.RandomState(1234)
        means = [float(random.choice(values, len(values), replace=True).mean()) for _ in range(1000)]
        result['image_bootstrap_95pct_m'] = np.percentile(means, [2.5, 97.5]).tolist()
    return result


def write_hierarchy_summary(directory, records, elapsed):
    rows = [r for r in records if r['type'] == 'hierarchy_result']
    summary = {'policies': {}, 'comparisons': []}
    for policy in POLICIES:
        selected = [r for r in rows if r['policy'] == policy]
        if not selected:
            continue
        summary['policies'][policy] = dict(
            metrics=pool_metrics([r['metrics'] for r in selected]),
            by_seed={str(seed): pool_metrics([r['metrics'] for r in selected if r['seed'] == seed])
                     for seed in sorted({r['seed'] for r in selected})},
            groups={name: pool_metrics([r['groups'][name] for r in selected]) for name in selected[0]['groups']},
            mean_objective=float(np.mean([r['losses']['objective'] for r in selected])))
    pairs = [('B_r0_only', 'A_keep_all'), ('C_learned', 'A_keep_all'),
             ('E_oracle_hierarchy', 'C_learned')] + [
                 (p, 'B_r0_only') for p in POLICIES if p not in ('A_keep_all', 'B_r0_only')]
    pairs += [('E_oracle_hierarchy', 'D_oracle_r{}'.format(r)) for r in range(1, 5)]
    summary['comparisons'] = [paired_comparison(rows, p, ref) for p, ref in pairs]
    (directory / 'hierarchy_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = ['# Hierarchical rejection: A--E experiment', '',
             'Checkpoint-only controlled tests; no weights updated. r0 decisions are identical in B--E.',
             'A: keep all. B: learned r0 only. C: learned at all levels.',
             'D_r: GT-guided rejection only at coarse level r + learned r0.',
             'E: GT-guided rejection at r1--r4 + learned r0.',
             'Oracle: reject known bad; restore known good; keep ambiguous/unknown measurements.', '',
             '## Final metrics (pixel-weighted across images and seeds)', '',
             '| Policy | MAE mm | RMSE mm | iMAE 1/km | iRMSE 1/km |', '|---|---:|---:|---:|---:|']
    def number(value, factor=1):
        return 'unavailable' if value is None else '{:.3f}'.format(value * factor)
    for policy in POLICIES:
        if policy in summary['policies']:
            m = summary['policies'][policy]['metrics']
            lines.append('| {} | {} | {} | {} | {} |'.format(policy,
                number(m.get('mae_m'), 1000), number(m.get('rmse_m'), 1000),
                number(m.get('imae_per_km')), number(m.get('irmse_per_km'))))
    lines += ['', '## Paired image comparisons', '',
              'Negative differences favor the first policy. CI resamples images after averaging seeds.',
              'This is a mean of per-image RMSE differences, not a difference of pooled RMSE.', '',
              '| Policy vs reference | Mean delta mm | 95% image-bootstrap CI mm | Images improved / tested |',
              '|---|---:|---|---:|']
    for comparison in summary['comparisons']:
        if not comparison['n_images']:
            continue
        interval = comparison.get('image_bootstrap_95pct_m')
        lines.append('| {} vs {} | {} | {} | {} / {} |'.format(
            comparison['policy'], comparison['reference'], number(comparison['mean_image_rmse_delta_m'], 1000),
            'not enough images' if interval is None else '[{}, {}]'.format(number(interval[0], 1000), number(interval[1], 1000)),
            comparison['images_improved'], comparison['n_images']))
    lines += ['', '## Interpretation', '',
              '- C vs B measures the current learned coarse hierarchy; E vs B tests corrected coarse decisions.',
              '- D vs B localizes useful levels. E vs each D tests whether combining levels adds value.',
              '- B vs A tests the reused learned r0 mask; this is not retrained r0-only performance.',
              '- Per-image JSONL includes rejection confusion counts, stage changes, SSE added/removed, depth and boundary groups.',
              '- Masks are frozen, but geometry, propagation weights and refinement recompute normally.',
              '- The oracle is measurement-consistency-based, not an optimal-mask upper bound.',
              '- GT-assisted results are not deployable scores. Improvements do not prove a head can learn the oracle.',
              '- Missing/ambiguous GT limits the oracle. Do not tune thresholds on this evaluation set.',
              '- Small subsets and one checkpoint cannot establish generalization; confirm promising results on the full set.',
              '- Stage errors use unclamped depth; final metrics use the model output and normal evaluation bounds.',
              '', 'Elapsed: {:.1f} seconds. Share this directory\'s sibling ZIP.'.format(elapsed)]
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
