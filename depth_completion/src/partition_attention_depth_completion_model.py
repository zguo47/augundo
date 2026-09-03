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
                 device=torch.device('cuda')):
        del dataset_name, network_modules

        self.model_depth = PartitionAttentionDepthModel(
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth)

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
        del intrinsics, validity_map

        output_depth = self.model_depth(
            image=image,
            sparse_depth=sparse_depth)

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
