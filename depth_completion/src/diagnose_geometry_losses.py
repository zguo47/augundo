'''Measure learned geometry separately from propagated metric predictions.

Run the same image indices/checkpoint age for SI, affine and aligned-L1 runs.
Every checkpoint is scored with ALL geometry measures, regardless of the loss
that trained it. A low affine error alone can hide inaccurate depth ratios.
'''
import json
import numpy as np
import torch
from diagnose_scale_propagation import error_metrics, seed_everything
from diagnose_hierarchical_rejection import pool_metrics
from utils.src import loss_utils


@torch.no_grad()
def diagnose_sample(wrapper, image, sparse, target, args, sample, directory, emit, save_maps):
    gt = target.to(wrapper.device)
    valid = torch.isfinite(gt) & (gt > max(0., args.min_evaluate_depth)) & (gt < args.max_evaluate_depth)
    for seed in args.seeds:
        seed_everything((seed + 1000003 * sample) % (2**32))
        outputs = wrapper.forward_depth(image, sparse, return_all_outputs=True)
        _, trained_losses = wrapper.compute_loss_supervised(gt, outputs, {
            'w_si': args.w_si, 'w_log_l1': args.w_log_l1, 'w_affine': args.w_affine,
            'w_scale_aligned_l1': args.w_scale_aligned_l1, 'w_supervised': args.w_supervised},
            intrinsics=wrapper.loss_intrinsics)
        emit('geometry_final', sample=sample, seed=seed,
             metrics=error_metrics(outputs[0], gt, valid),
             losses={k: float(v) for k,v in trained_losses.items()})
        maps = dict(gt=target.numpy(), sparse=sparse.cpu().numpy(), final=outputs[0].cpu().numpy())
        for level, z in enumerate(outputs[1:6]):
            si = loss_utils.scale_invariant_loss_func(z, gt, valid.float())
            aligned_l1, a = loss_utils.scale_aligned_l1(z, gt, valid)
            affine, aa, b = loss_utils.affine_invariant_loss(z, gt, valid, wrapper.loss_intrinsics)
            aligned = a[:, None, None, None] * z
            affine_z = aa[:, None, None, None] * z + b[:, None, None, None]
            # Local log-ratio errors use only pairs with two valid GT pixels.
            # Never interpolate labels or turn missing values into surfaces.
            log_error = z.clamp_min(1e-7).log() - gt.clamp_min(1e-7).log()
            pair_stats = {}
            for distance in (1, 2, 4, 8):
                errors = []
                for axis in (-2, -1):
                    if z.shape[axis] <= distance:
                        continue
                    left = [slice(None)] * 4; right = list(left)
                    left[axis], right[axis] = slice(None, -distance), slice(distance, None)
                    left, right = tuple(left), tuple(right)
                    pair = valid[left] & valid[right]
                    errors.append((log_error[left] - log_error[right]).abs()[pair])
                values = torch.cat(errors) if errors else z.new_empty(0)
                pair_stats[str(distance)] = dict(count=values.numel(), mae=float(values.mean()) if values.numel() else None)
            emit('geometry_level', sample=sample, seed=seed, level=level,
                 si=float(si), aligned_l1=float(aligned_l1), affine_loss=float(affine),
                 alignment_scale=a.cpu().tolist(), affine_scale=aa.cpu().tolist(), affine_shift=b.cpu().tolist(),
                 affine_nonpositive_fraction=float((affine_z[valid] <= 0).float().mean()) if valid.any() else None,
                 metrics=error_metrics(aligned, gt, valid),
                 affine_depth_metrics=error_metrics(affine_z, gt, valid & (aa[:, None, None, None] > 0)),
                 far_metrics=error_metrics(aligned, gt, valid & (gt > 40)), pairs=pair_stats)
            maps['r{}_aligned'.format(level)] = aligned.cpu().numpy()
            maps['r{}_affine_aligned'.format(level)] = affine_z.cpu().numpy()
        if save_maps and seed == args.seeds[0]:
            np.savez_compressed(str(directory / 'geometry_{}_{}.npz'.format(sample, seed)), **maps)
        print('geometry sample {}, seed {} complete'.format(sample, seed), flush=True)


def write_summary(directory, records):
    final = [r for r in records if r['type'] == 'geometry_final']
    result = dict(final=pool_metrics([r['metrics'] for r in final]), levels={})
    def millimetres(metric):
        value = metric.get('rmse_m')
        return 'unavailable' if value is None else '{:.3f}'.format(value * 1000)
    lines = ['# Geometry loss experiment', '',
             'Compare checkpoints from matched training runs and the SAME indices/seeds.',
             'All checkpoints are scored with all losses. Affine alignment is evaluation only.', '',
             'Final prediction RMSE: {} mm.'.format(millimetres(result['final'])), '',
             '| Level | Scale-aligned RMSE mm | Affine-aligned z RMSE mm | SI | Affine loss |',
             '|---|---:|---:|---:|---:|']
    for level in range(5):
        rows = [r for r in records if r['type'] == 'geometry_level' and r['level'] == level]
        if not rows:
            continue
        entry = dict(metrics=pool_metrics([r['metrics'] for r in rows]),
                     affine_depth_metrics=pool_metrics([r['affine_depth_metrics'] for r in rows]),
                     far_metrics=pool_metrics([r['far_metrics'] for r in rows]),
                     mean_si=float(np.mean([r['si'] for r in rows])),
                     mean_affine_loss=float(np.mean([r['affine_loss'] for r in rows])))
        entry['pairs'] = {}
        for distance in ('1', '2', '4', '8'):
            pairs = [r['pairs'][distance] for r in rows if r['pairs'][distance]['count']]
            count = sum(p['count'] for p in pairs)
            entry['pairs'][distance] = dict(count=count,
                mae=sum(p['mae'] * p['count'] for p in pairs) / count if count else None)
        entry['positive_affine_fits'] = sum(sum(a > 0 for a in r['affine_scale']) for r in rows)
        result['levels']['r{}'.format(level)] = entry
        lines.append('| {} | {} | {} | {:.6f} | {:.6f} |'.format(level,
            millimetres(entry['metrics']),
            millimetres(entry['affine_depth_metrics']),
            entry['mean_si'], entry['mean_affine_loss']))
    lines += ['', 'Inspect JSONL for per-image scales, offsets, nonpositive aligned depths, far-depth errors and local log-ratio errors.',
              'Success requires improved scale-only geometry and final depth, not merely a lower affine objective.',
              'The affine objective uses XYZ and inverse-depth weighting; its z-only RMSE is a separate diagnostic.',
              'No loss or GT alignment is applied to the actual final prediction.']
    (directory / 'geometry_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
