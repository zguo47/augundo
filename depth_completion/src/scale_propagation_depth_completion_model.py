import torch
import torch.nn as nn
import torch.nn.functional as functional

from scale_propagation_model import ScalePropagationDepthModel
from utils.src import loss_utils


class ScalePropagationDepthCompletionModel(object):
    '''Repository wrapper for shared-encoder multi-scale propagation.'''

    def __init__(self,
                 dataset_name='kitti',
                 network_modules=None,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 use_outlier_rejection=True,
                 device=torch.device('cuda'),
                 detach_scaleless_depth=True):
        del dataset_name, network_modules

        self.model_depth = ScalePropagationDepthModel(
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            use_outlier_rejection=use_outlier_rejection,
            detach_scaleless_depth=detach_scaleless_depth)

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.device = device
        self.to(device)

    def forward_depth(self,
                      image,
                      sparse_depth=None,
                      validity_map=None,
                      intrinsics=None,
                      return_all_outputs=False):
        # Keep the shared repository call signature; this model derives sparse
        # measurement locations directly from positive depths at each level.
        del validity_map, intrinsics

        output_depth = self.model_depth(
            image=image,
            sparse_depth=sparse_depth,
            return_all_outputs=return_all_outputs)
        if not return_all_outputs:
            return output_depth
        # Flat, full-resolution tensors follow the repository's transform API.
        # Reverse geometric augmentation on every map before computing losses.
        # Layout: final depth, five Z maps, five M maps, five metric-loss maps.
        maps = output_depth['scaleless_depths'] + output_depth['scale_maps'] + \
            output_depth['metric_depths_for_loss']
        return [output_depth['depth']] + [
            functional.interpolate(
                value, size=image.shape[-2:], mode='bilinear', align_corners=True)
            for value in maps]

    @staticmethod
    def scale_outputs(outputs):
        '''Extract aligned maps from the explicit full-resolution output list.'''
        return dict(scaleless_depths=outputs[1:6], scale_maps=outputs[6:11])

    def compute_loss_supervised(self, target_depth, output_depth, w_losses):
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
        w_supervised = w_losses.get('w_supervised', 1.0)
        loss = w_supervised * (loss_si + loss_log_l1)

        loss_info.update({
            'loss': loss,
            'loss_si': loss_si,
            'loss_log_l1': loss_log_l1
        })
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
