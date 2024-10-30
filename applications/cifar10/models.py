"""
Code from https://github.com/andrew-cr/tauLDR/blob/main/lib/models/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from applications.cifar10 import networks
from applications.cifar10 import model_utils
from applications.cifar10.forward_processes import GaussianTargetRateForwardProcess


def count_params(model):
    return sum([p.numel() for p in model.parameters()])


class ImageX0PredBase(nn.Module):
    def __init__(self, cfg, device=None, rank=None):
        """
        Args:
            device: Allows the option to load a trained model on a different device
                than that used for training
        """
        super().__init__()

        self.S = cfg.data.S
        self.data_shape = cfg.data.shape
        self.fix_logistic = cfg.model.fix_logistic

        if device:
            self.device = device
        else:
            self.device = cfg.device

        ch = cfg.model.ch
        num_res_blocks = cfg.model.num_res_blocks
        num_scales = cfg.model.num_scales
        ch_mult = cfg.model.ch_mult
        input_channels = cfg.model.input_channels
        # output_channels not actually used
        output_channels = cfg.model.input_channels * cfg.data.S
        scale_count_to_put_attn = cfg.model.scale_count_to_put_attn
        data_min_max = cfg.model.data_min_max
        dropout = cfg.model.dropout
        skip_rescale = cfg.model.skip_rescale
        do_time_embed = True
        time_scale_factor = cfg.model.time_scale_factor
        time_embed_dim = cfg.model.time_embed_dim

        tmp_net = networks.UNet(
            ch,
            num_res_blocks,
            num_scales,
            ch_mult,
            input_channels,
            output_channels,
            scale_count_to_put_attn,
            data_min_max,
            dropout,
            skip_rescale,
            do_time_embed,
            time_scale_factor,
            time_embed_dim,
        ).to(self.device)
        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

    def forward(self, x, times):
        """
        Returns logits over state space for each pixel
        """
        x = x["x"]
        B, D = x.shape
        C, H, W = self.data_shape
        S = self.S
        x = x.view(B, C, H, W)

        net_out = self.net(x, times)  # (B, 2*C, H, W)

        # Truncated logistic output from https://arxiv.org/pdf/2107.03006.pdf

        mu = net_out[:, 0:C, :, :].unsqueeze(-1)
        log_scale = net_out[:, C:, :, :].unsqueeze(-1)

        inv_scale = torch.exp(-(log_scale - 2))

        bin_width = 2.0 / self.S
        bin_centers = torch.linspace(
            start=-1.0 + bin_width / 2,
            end=1.0 - bin_width / 2,
            steps=self.S,
            device=self.device,
        ).view(1, 1, 1, 1, self.S)

        sig_in_left = (bin_centers - bin_width / 2 - mu) * inv_scale
        bin_left_logcdf = F.logsigmoid(sig_in_left)
        sig_in_right = (bin_centers + bin_width / 2 - mu) * inv_scale
        bin_right_logcdf = F.logsigmoid(sig_in_right)

        logits_1 = self._log_minus_exp(bin_right_logcdf, bin_left_logcdf)
        logits_2 = self._log_minus_exp(
            -sig_in_left + bin_left_logcdf, -sig_in_right + bin_right_logcdf
        )
        if self.fix_logistic:
            logits = torch.min(logits_1, logits_2)
        else:
            logits = logits_1

        logits = logits.view(B, D, S)

        return logits

    def _log_minus_exp(self, a, b, eps=1e-6):
        """
        Compute log (exp(a) - exp(b)) for (b<a)
        From https://arxiv.org/pdf/2107.03006.pdf
        """
        return a + torch.log1p(-torch.exp(b - a) + eps)


# Based on https://github.com/yang-song/score_sde_pytorch/blob/ef5cb679a4897a40d20e94d8d0e2124c3a48fb8c/models/ema.py
class EMA:
    def __init__(self, cfg):
        self.decay = cfg.model.ema_decay
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.shadow_params = []
        self.collected_params = []
        self.num_updates = 0

    def init_ema(self):
        self.shadow_params = [
            p.clone().detach() for p in self.parameters() if p.requires_grad
        ]

    def update_ema(self):

        if len(self.shadow_params) == 0:
            raise ValueError("Shadow params not initialized before first ema update!")

        decay = self.decay
        self.num_updates += 1
        decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in self.parameters() if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def state_dict(self):
        sd = nn.Module.state_dict(self)
        sd["ema_decay"] = self.decay
        sd["ema_num_updates"] = self.num_updates
        sd["ema_shadow_params"] = self.shadow_params

        return sd

    def move_shadow_params_to_model_params(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def move_model_params_to_collected_params(self):
        self.collected_params = [param.clone() for param in self.parameters()]

    def move_collected_params_to_model_params(self):
        for c_param, param in zip(self.collected_params, self.parameters()):
            param.data.copy_(c_param.data)

    def load_state_dict(self, state_dict):
        missing_keys, unexpected_keys = nn.Module.load_state_dict(
            self, state_dict, strict=False
        )

        # print("state dict keys")
        # for key in state_dict.keys():
        #     print(key)

        if len(missing_keys) > 0:
            print("Missing keys: ", missing_keys)
            raise ValueError
        if not (
            len(unexpected_keys) == 3
            and "ema_decay" in unexpected_keys
            and "ema_num_updates" in unexpected_keys
            and "ema_shadow_params" in unexpected_keys
        ):
            print("Unexpected keys: ", unexpected_keys)
            raise ValueError

        self.decay = state_dict["ema_decay"]
        self.num_updates = state_dict["ema_num_updates"]
        self.shadow_params = state_dict["ema_shadow_params"]

    def train(self, mode=True):
        if self.training == mode:
            print(
                "Dont call model.train() with the same mode twice! Otherwise EMA parameters may overwrite original parameters"
            )
            print("Current model training mode: ", self.training)
            print("Requested training mode: ", mode)
            raise ValueError

        nn.Module.train(self, mode)
        if mode:
            if len(self.collected_params) > 0:
                self.move_collected_params_to_model_params()
            else:
                print("model.train(True) called but no ema collected parameters!")
        else:
            self.move_model_params_to_collected_params()
            self.move_shadow_params_to_model_params()


# This class is used for initializing the model ckpt trained by campbell
# make sure EMA inherited first so it can override the state dict functions
@model_utils.register_model
class GaussianTargetRateImageX0PredEMA(
    EMA, ImageX0PredBase, GaussianTargetRateForwardProcess
):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBase.__init__(self, cfg, device, rank)
        GaussianTargetRateForwardProcess.__init__(self, cfg, device)

        self.init_ema()
