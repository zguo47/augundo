import time

import torch
import torch.nn as nn

from partition_attention_model import PartitionAttentionDepthModel
from utils.src import loss_utils


class PartitionAttentionDepthCompletionModel(object):
    '''Repository wrapper for the minimal RGB/sparse partition-attention model.'''

    def __init__(self,
                 dataset_name='kitti',
                 network_modules=None,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 n_iteration=1,
                 window_size=16,
                 n_self_attention=2,
                 n_shift=4,
                 device=torch.device('cuda')):
        del dataset_name, network_modules

        self.model_depth = PartitionAttentionDepthModel(
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            n_iteration=n_iteration,
            window_size=window_size,
            n_self_attention=n_self_attention,
            n_shift=n_shift)

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.print_inference_time = False
        self.device = device
        self.to(device)

    def forward_depth(self,
                      image,
                      sparse_depth=None,
                      validity_map=None,
                      intrinsics=None,
                      return_all_outputs=False):
        del intrinsics

        time_inference = self.print_inference_time and not self.model_depth.training
        if time_inference:
            if image.is_cuda:
                torch.cuda.synchronize()
            time_start = time.time()

        output_depth = self.model_depth(
            image=image)

        if time_inference:
            if image.is_cuda:
                torch.cuda.synchronize()
            seconds_per_image = \
                (time.time() - time_start) / image.shape[0]
            print('Inference time: {:.4f} s/image'.format(seconds_per_image))

        return [output_depth] if return_all_outputs else output_depth

    def compute_loss_supervised(self, target_depth, output_depth, w_losses):
        '''Computes metric log-L1 loss over valid target pixels.'''
        output_depth = output_depth[0]

        validity = (target_depth > 0.0).to(target_depth.dtype)
        target_depth = torch.where(
            validity > 0.0,
            torch.clamp(
                target_depth,
                min=self.min_predict_depth,
                max=self.max_predict_depth),
            torch.full_like(target_depth, self.min_predict_depth))

        loss_log_l1 = loss_utils.log_l1_loss_func(
            src=output_depth,
            tgt=target_depth,
            w=validity)
        w_supervised = w_losses.get('w_supervised', 1.0)
        loss = w_supervised * loss_log_l1

        return loss, {
            'loss': loss,
            'loss_log_l1': loss_log_l1
        }

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
        '''Saves model and optional optimizer state.'''
        checkpoint = {
            'train_step': step,
            'model_state_dict': self._model_state_dict()
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, restore_path, optimizer=None):
        '''Restores model and optimizer state when one is available.'''
        checkpoint = torch.load(restore_path, map_location=self.device)
        model = self.model_depth.module \
            if isinstance(self.model_depth, nn.DataParallel) \
            else self.model_depth
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint.get('train_step', 0), optimizer
