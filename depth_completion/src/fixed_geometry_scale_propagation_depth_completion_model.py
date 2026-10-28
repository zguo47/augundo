import torch
import torch.nn as nn
import torch.nn.functional as functional

from fixed_geometry_scale_propagation_model import FixedGeometryScalePropagationDepthModel
from scale_propagation_depth_completion_model import ScalePropagationDepthCompletionModel
from utils.src import loss_utils


class FixedGeometryScalePropagationDepthCompletionModel(ScalePropagationDepthCompletionModel):
    '''Repository wrapper for the independent fixed-geometry baseline.

    Reuse only the repository-facing map layout, device methods, and common
    call signatures. The network, stage-specific losses, and checkpoint
    initialization below are separate from the current scale-propagation model.
    '''

    def __init__(self,
                 dataset_name='kitti',
                 network_modules=None,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda'),
                 n_propagation_iterations=(2, 4, 6, 8, 12),
                 encoder_type='conv',
                 training_stage='joint',
                 propagation_mode='hierarchical',
                 geometry_restore_path=None,
                 detach_scaleless_depth=True):
        del dataset_name, network_modules
        self.model_depth = FixedGeometryScalePropagationDepthModel(
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            n_propagation_iterations=n_propagation_iterations,
            encoder_type=encoder_type,
            training_stage=training_stage,
            propagation_mode=propagation_mode,
            detach_scaleless_depth=detach_scaleless_depth)
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.training_stage = training_stage
        self.geometry_restore_path = geometry_restore_path
        self.device = device
        self.to(device)
        if geometry_restore_path:
            # Start a NEW scale experiment: load geometry only, not the first
            # stage's optimizer, training step, or unused propagation weights.
            checkpoint = torch.load(geometry_restore_path, map_location=device)
            self.model_depth.geometry.load_state_dict(checkpoint['geometry_state_dict'])

    def forward_depth(self,
                      image,
                      sparse_depth=None,
                      validity_map=None,
                      intrinsics=None,
                      return_all_outputs=False,
                      return_rejection_outputs=False,
                      return_rejection_masks=False):
        del validity_map, intrinsics
        output = self.model_depth(
            image=image, sparse_depth=sparse_depth,
            return_all_outputs=return_all_outputs,
            return_rejection_outputs=return_rejection_outputs or return_rejection_masks)
        if not return_all_outputs and not return_rejection_outputs and not return_rejection_masks:
            return output

        # Preserve the original 16 entries for metric supervision, TensorBoard,
        # and diagnostics. Append the four auxiliary RGB geometries separately
        # so they cannot be mistaken for the geometry used by scale propagation.
        # All maps follow the same reverse-augmentation path before the loss.
        maps = output['scaleless_depths'] + output['scale_maps'] + \
            output['metric_depths_for_loss'] + output['auxiliary_scaleless_depths'][1:]
        outputs = [output['depth']] + [
            functional.interpolate(
                value, size=image.shape[-2:], mode='bilinear', align_corners=True)
            for value in maps]
        if return_rejection_masks:
            return outputs, output['rejection_keep_masks']
        if return_rejection_outputs:
            return outputs, output['rejection_logits']
        return outputs

    def compute_loss_supervised(self, target_depth, output_depth, w_losses):
        '''Train SI geometry and metric scale together, unless a stage is selected.'''
        validity = (target_depth > 0.0).to(target_depth.dtype)
        target_depth = torch.where(
            validity > 0.0,
            target_depth.clamp(self.min_predict_depth, self.max_predict_depth),
            torch.full_like(target_depth, self.min_predict_depth))

        # r0 is the final geometry; r1-r4 are independent auxiliary heads.
        # Each was upsampled to full resolution by forward_depth, avoiding
        # downsampling or pooling the ground truth. Propagation still uses
        # only the resized r0 maps in output_depth[1:6].
        loss_si = target_depth.new_zeros(())
        loss_log_l1 = target_depth.new_zeros(())
        loss_info = {}
        for resolution, geometry in enumerate([output_depth[1]] + output_depth[16:20]):
            value = loss_utils.scale_invariant_loss_func(
                src=geometry, tgt=target_depth, w=validity)
            loss_si = loss_si + value / (4 ** resolution)
            loss_info['loss_si_r{}'.format(resolution)] = value
        for resolution, metric in enumerate(output_depth[11:16]):
            value = loss_utils.log_l1_loss_func(src=metric, tgt=target_depth, w=validity)
            loss_log_l1 = loss_log_l1 + value / (4 ** resolution)
            loss_info['loss_log_l1_r{}'.format(resolution)] = value

        w_supervised = w_losses.get('w_supervised', 1.0)
        # Joint mode combines both objectives in the same optimizer step.
        # Separate-stage ablations still report, but do not train, the other.
        loss_si_weighted = w_supervised * w_losses.get('w_si', 1.0) * loss_si \
            if self.training_stage in ['joint', 'geometry'] else loss_si.detach() * 0.0
        loss_log_l1_weighted = w_supervised * w_losses.get('w_log_l1', 1.0) * loss_log_l1 \
            if self.training_stage in ['joint', 'scale'] else loss_log_l1.detach() * 0.0
        loss = loss_si_weighted + loss_log_l1_weighted
        loss_info.update(
            loss=loss, loss_si=loss_si, loss_log_l1=loss_log_l1,
            loss_si_weighted=loss_si_weighted,
            loss_log_l1_weighted=loss_log_l1_weighted)
        return loss, loss_info

    def parameters(self):
        # Adam receives only the stage's trainable parameters. In particular,
        # frozen geometry cannot change through weight decay or optimizer state.
        return [parameter for parameter in self.model_depth.parameters()
                if parameter.requires_grad]

    def save_model(self, checkpoint_path, step, optimizer=None):
        model = self.model_depth.module \
            if isinstance(self.model_depth, nn.DataParallel) else self.model_depth
        checkpoint = dict(
            train_step=step,
            model_state_dict=model.state_dict(),
            geometry_state_dict=model.geometry.state_dict(),
            fixed_geometry_stage=self.training_stage)
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, checkpoint_path)
