import torch
import torch.nn as nn
import torch.nn.functional as functional

from scale_propagation_model import ScalePropagationDepthModel, sparse_depth_at_resolution
from utils.src import loss_utils


class ScalePropagationDepthCompletionModel(object):
    '''Repository wrapper for shared-encoder multi-scale propagation.'''

    accepts_geometry_intrinsics = True

    def __init__(self,
                 dataset_name='kitti',
                 network_modules=None,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 use_outlier_rejection=True,
                 device=torch.device('cuda'),
                 detach_scaleless_depth=True,
                 n_propagation_iterations=(2, 4, 6, 8, 12),
                 n_scale_refinement=2,
                 propagation_anchor_strength=2.0,
                 encoder_type='conv',
                 use_sparse_depth_features=False):
        del dataset_name, network_modules

        self.model_depth = ScalePropagationDepthModel(
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            use_outlier_rejection=use_outlier_rejection,
            detach_scaleless_depth=detach_scaleless_depth,
            n_propagation_iterations=n_propagation_iterations,
            n_scale_refinement=n_scale_refinement,
            propagation_anchor_strength=propagation_anchor_strength,
            encoder_type=encoder_type,
            use_sparse_depth_features=use_sparse_depth_features)

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.device = device
        self.to(device)

    def forward_depth(self,
                      image,
                      sparse_depth=None,
                      validity_map=None,
                      intrinsics=None,
                      return_all_outputs=False,
                      return_rejection_outputs=False,
                      return_rejection_masks=False):
        # Keep the shared repository call signature; this model derives sparse
        # measurement locations directly from positive depths at each level.
        del validity_map, intrinsics

        output_depth = self.model_depth(
            image=image,
            sparse_depth=sparse_depth,
            return_all_outputs=return_all_outputs,
            return_rejection_outputs=return_rejection_outputs or return_rejection_masks)
        if not return_all_outputs and not return_rejection_outputs and not return_rejection_masks:
            return output_depth
        # Flat, full-resolution tensors follow the repository's transform API.
        # Reverse geometric augmentation on every map before computing losses.
        # Layout: final depth, five Z maps, five M maps, five metric-loss maps.
        maps = output_depth['scaleless_depths'] + output_depth['scale_maps'] + \
            output_depth['metric_depths_for_loss']
        outputs = [output_depth['depth']] + [
            functional.interpolate(
                value, size=image.shape[-2:], mode='bilinear', align_corners=True)
            for value in maps]
        if return_rejection_masks:
            # These are the exact masks used for anchor injection in this pass.
            return outputs, output_depth['rejection_keep_masks']
        if return_rejection_outputs:
            # Keep native logits separate from the image/map transform API:
            # their labels are computed in the augmented input coordinates.
            return outputs, output_depth['rejection_logits']
        return outputs

    def compute_loss_rejection(self, target_depth, sparse_depth, logits,
                               keep_tolerance=(0.1, 0.02),
                               reject_tolerance=(0.3, 0.05),
                               supervision_levels=None,
                               loss_type='cross_entropy',
                               keep_weight=2.0):
        '''Sum resolution-weighted classification or rejection-sparsity losses.'''
        if supervision_levels is None:
            supervision_levels = range(5)
        if loss_type == 'none':
            loss = logits[0].sum() * 0.0
            return loss, {'loss_rejection': loss}
        losses, loss_info = [], {}
        for resolution, scores in enumerate(logits):
            # The same level selector applies to both objectives. In L1 mode,
            # scores is a sampled keep mask rather than two-channel logits.
            if resolution not in supervision_levels:
                continue
            selected, indices = sparse_depth_at_resolution(
                sparse_depth, scores.shape[-2:], return_indices=True)
            if loss_type == 'l1':
                value = loss_utils.rejection_sparsity_loss_func(scores, selected)
                loss_info['rejection_kept_fraction_r{}'.format(resolution)] = value.detach()
                measured = (selected > 0.0).to(scores.dtype)
                rejected_fraction = (measured * (1.0 - scores)).sum() / measured.sum().clamp_min(1)
                loss_info['rejection_rejected_fraction_r{}'.format(resolution)] = rejected_fraction.detach()
            else:
                matched = torch.gather(target_depth.flatten(2), 2, indices.flatten(2)).reshape_as(selected)
                labels = loss_utils.rejection_supervision_targets(
                    selected, matched, keep_tolerance, reject_tolerance)
                value = loss_utils.balanced_rejection_loss_func(scores, labels, keep_weight)
                for name, label in [('keep', 0), ('reject', 1), ('ignored', -1)]:
                    loss_info['rejection_{}_count_r{}'.format(name, resolution)] = (labels == label).sum()
            weight = 1.0 / (4 ** resolution)
            losses.append(weight * value)
            loss_info['loss_rejection_r{}'.format(resolution)] = value
            loss_info['loss_rejection_weighted_r{}'.format(resolution)] = weight * value
        # Match SI and metric losses: sum 4**(-r) times each selected loss.
        # Both CE and L1 use this weighting, with r=0 at full resolution.
        loss = torch.stack(losses).sum()
        loss_info['loss_rejection'] = loss
        return loss, loss_info

    @staticmethod
    def scale_outputs(outputs):
        '''Extract aligned maps from the explicit full-resolution output list.'''
        return dict(scaleless_depths=outputs[1:6], scale_maps=outputs[6:11])

    def compute_loss_supervised(self, target_depth, output_depth, w_losses, intrinsics=None):
        '''Supervise all five geometries and metric products, including bottom.'''

        validity = (target_depth > 0.0).to(target_depth.dtype)
        target_depth = torch.where(
            validity > 0.0,
            torch.clamp(
                target_depth,
                min=self.min_predict_depth,
                max=self.max_predict_depth),
            torch.full_like(target_depth, self.min_predict_depth))

        loss_si = target_depth.new_zeros(())
        loss_log_l1 = target_depth.new_zeros(())
        loss_affine = target_depth.new_zeros(())
        loss_aligned_l1 = target_depth.new_zeros(())
        loss_info = {}
        for resolution, (geometry, metric) in enumerate(zip(
                output_depth[1:6], output_depth[11:16])):
            # Metric products optionally detach geometry; SI supervises
            # the geometrically aligned scaleless prediction directly.
            scale_invariant = loss_utils.scale_invariant_loss_func(
                src=geometry, tgt=target_depth, w=validity)
            metric_loss = loss_utils.log_l1_loss_func(
                src=metric, tgt=target_depth, w=validity)
            weight = 1.0 / (4 ** resolution)
            loss_si = loss_si + weight * scale_invariant
            loss_log_l1 = loss_log_l1 + weight * metric_loss
            loss_info['loss_si_r{}'.format(resolution)] = scale_invariant
            loss_info['loss_log_l1_r{}'.format(resolution)] = metric_loss
            # Optional geometry experiments use the same five-level weights.
            # SI can remain active or be disabled with w_si=0 for replacement.
            if w_losses.get('w_scale_aligned_l1', 0.0) != 0:
                aligned, alignment = loss_utils.scale_aligned_l1(geometry, target_depth, validity)
                loss_aligned_l1 = loss_aligned_l1 + weight * aligned
                loss_info['loss_scale_aligned_l1_r{}'.format(resolution)] = aligned
                loss_info['geometry_alignment_scale_r{}'.format(resolution)] = alignment.detach().mean()
            if w_losses.get('w_affine', 0.0) != 0:
                affine, alignment, shift = loss_utils.affine_invariant_loss(
                    geometry, target_depth, validity, intrinsics)
                loss_affine = loss_affine + weight * affine
                loss_info['loss_affine_r{}'.format(resolution)] = affine
                loss_info['geometry_affine_scale_r{}'.format(resolution)] = alignment.detach().mean()
                loss_info['geometry_affine_shift_r{}'.format(resolution)] = shift.detach().mean()
                loss_info['geometry_affine_valid_r{}'.format(resolution)] = (alignment.detach() > 0).float().mean()
        w_supervised = w_losses.get('w_supervised', 1.0)
        w_si = w_losses.get('w_si', 1.0)
        w_log_l1 = w_losses.get('w_log_l1', 1.0)
        loss_si_weighted = w_supervised * w_si * loss_si
        loss_log_l1_weighted = w_supervised * w_log_l1 * loss_log_l1
        loss = loss_si_weighted + loss_log_l1_weighted
        loss_affine_weighted = w_supervised * w_losses.get('w_affine', 0.0) * loss_affine
        loss_aligned_l1_weighted = w_supervised * w_losses.get('w_scale_aligned_l1', 0.0) * loss_aligned_l1
        loss = loss + loss_affine_weighted + loss_aligned_l1_weighted

        loss_info.update({
            'loss': loss,
            'loss_si': loss_si,
            'loss_log_l1': loss_log_l1,
            'loss_si_weighted': loss_si_weighted,
            'loss_log_l1_weighted': loss_log_l1_weighted
        })
        loss_info.update(loss_affine=loss_affine, loss_affine_weighted=loss_affine_weighted,
                         loss_scale_aligned_l1=loss_aligned_l1,
                         loss_scale_aligned_l1_weighted=loss_aligned_l1_weighted)
        return loss, loss_info

    def parameters(self):
        return list(self.model_depth.parameters())

    def parameters_depth(self):
        return self.parameters()

    def parameters_pose(self):
        return []

    def forward_pose(self, image0, image1):
        del image0, image1
        return None

    def train(self):
        self.model_depth.train()

    def eval(self):
        self.model_depth.eval()

    def to(self, device):
        self.device = device
        self.model_depth.to(device)

    def data_parallel(self):
        self.model_depth = nn.DataParallel(self.model_depth)

    def _model_state_dict(self):
        model = self.model_depth.module \
            if isinstance(self.model_depth, nn.DataParallel) \
            else self.model_depth
        return model.state_dict()

    def save_model(self, checkpoint_path, step, optimizer=None):
        checkpoint = {
            'train_step': step,
            'model_state_dict': self._model_state_dict()
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, restore_path, optimizer=None):
        checkpoint = torch.load(restore_path, map_location=self.device)
        model = self.model_depth.module \
            if isinstance(self.model_depth, nn.DataParallel) \
            else self.model_depth
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint.get('train_step', 0), optimizer
