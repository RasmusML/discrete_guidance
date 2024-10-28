"""
Code from https://github.com/andrew-cr/tauLDR/blob/main/lib/networks/networks.py
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


# Code modified from https://github.com/yang-song/score_sde_pytorch
def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init


def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

class NiN(nn.Module):
  def __init__(self, in_ch, out_ch, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_ch, out_ch)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(out_ch), requires_grad=True)

  def forward(self, x, #  ["batch", "in_ch", "H", "W"]
    ):

    x = x.permute(0, 2, 3, 1)
    # x (batch, H, W, in_ch)
    y = torch.einsum('bhwi,ik->bhwk', x, self.W) + self.b
    # y (batch, H, W, out_ch)
    return y.permute(0, 3, 1, 2)

class AttnBlock(nn.Module):
  """Channel-wise self-attention block."""
  def __init__(self, channels, skip_rescale=True):
    super().__init__()
    self.skip_rescale = skip_rescale
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels//4, 32),
        num_channels=channels, eps=1e-6)
    self.NIN_0 = NiN(channels, channels)
    self.NIN_1 = NiN(channels, channels)
    self.NIN_2 = NiN(channels, channels)
    self.NIN_3 = NiN(channels, channels, init_scale=0.)

  def forward(self, x, # ["batch", "channels", "H", "W"]
    ):

    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.NIN_3(h)

    if self.skip_rescale:
        return (x + h) / np.sqrt(2.)
    else:
        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim=None, dropout=0.1, skip_rescale=True):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.skip_rescale = skip_rescale

        self.act = nn.functional.silu
        self.groupnorm0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32),
            num_channels=in_ch, eps=1e-6
        )
        self.conv0 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=1
        )

        if temb_dim is not None:
            self.dense0 = nn.Linear(temb_dim, out_ch)
            nn.init.zeros_(self.dense0.bias)


        self.groupnorm1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32),
            num_channels=out_ch, eps=1e-6
        )
        self.dropout0 = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1
        )
        if out_ch != in_ch:
            self.nin = NiN(in_ch, out_ch)

    def forward(self, x, # ["batch", "in_ch", "H", "W"]
                temb=None, #  ["batch", "temb_dim"]
        ):
        assert x.shape[1] == self.in_ch
        
        # Bear: groupnorm leading to error: 
        # RuntimeError: Expected memory formats of X and dY are same.
        # x = x.contiguous()
        # print(f'before groupnorm0: {x.is_contiguous()}')
        h = self.groupnorm0(x)
        # print(f'after groupnorm0: {h.is_contiguous()}')
        h = self.act(h)
        h = self.conv0(h)

        if temb is not None:
            h += self.dense0(self.act(temb))[:, :, None, None]

        h = self.groupnorm1(h)
        h = self.act(h)
        h = self.dropout0(h)
        h = self.conv1(h)
        if h.shape[1] != self.in_ch:
            x = self.nin(x)

        assert x.shape == h.shape

        if self.skip_rescale:
            return (x + h) / np.sqrt(2.)
        else:
            return x + h

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, 
            stride=2, padding=0)

    def forward(self, x, # ["batch", "ch", "inH", "inW"]
        ):
        B, C, H, W = x.shape
        x = nn.functional.pad(x, (0, 1, 0, 1))
        x= self.conv(x)

        assert x.shape == (B, C, H // 2, W // 2)
        return x

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, # ["batch", "ch", "inH", "inW"]
    ):
        B, C, H, W = x.shape
        h = F.interpolate(x, (H*2, W*2), mode='nearest')
        h = self.conv(h)

        assert h.shape == (B, C, H*2, W*2)
        return h

class UNet(nn.Module):
    def __init__(self, ch, num_res_blocks, num_scales, ch_mult, input_channels,
        output_channels, scale_count_to_put_attn, data_min_max, dropout,
        skip_rescale, do_time_embed, time_scale_factor=None, time_embed_dim=None):
        super().__init__()
        assert num_scales == len(ch_mult)

        self.ch = ch
        self.num_res_blocks = num_res_blocks
        self.num_scales = num_scales
        self.ch_mult = ch_mult
        self.input_channels = input_channels
        self.output_channels = 2 * input_channels
        self.scale_count_to_put_attn = scale_count_to_put_attn
        self.data_min_max = data_min_max # tuple of min and max value of input so it can be rescaled to [-1, 1]
        self.dropout = dropout
        self.skip_rescale = skip_rescale
        self.do_time_embed = do_time_embed # Whether to add in time embeddings
        self.time_scale_factor = time_scale_factor # scale to make the range of times be 0 to 1000
        self.time_embed_dim = time_embed_dim

        self.act = nn.functional.silu

        if self.do_time_embed:
            self.temb_modules = []
            self.temb_modules.append(nn.Linear(self.time_embed_dim, self.time_embed_dim*4))
            nn.init.zeros_(self.temb_modules[-1].bias)
            self.temb_modules.append(nn.Linear(self.time_embed_dim*4, self.time_embed_dim*4))
            nn.init.zeros_(self.temb_modules[-1].bias)
            self.temb_modules = nn.ModuleList(self.temb_modules)

        self.expanded_time_dim = 4 * self.time_embed_dim if self.do_time_embed else None

        self.input_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=self.ch,
            kernel_size=3, padding=1
        )

        h_cs = [self.ch]
        in_ch = self.ch


        # Downsampling
        self.downsampling_modules = []

        for scale_count in range(self.num_scales):
            for res_count in range(self.num_res_blocks):
                out_ch = self.ch * self.ch_mult[scale_count]
                self.downsampling_modules.append(
                    ResBlock(in_ch, out_ch, temb_dim=self.expanded_time_dim,
                        dropout=dropout, skip_rescale=self.skip_rescale)
                )
                in_ch = out_ch
                h_cs.append(in_ch)
                if scale_count == self.scale_count_to_put_attn:
                    self.downsampling_modules.append(
                        AttnBlock(in_ch, skip_rescale=self.skip_rescale)
                    )

            if scale_count != self.num_scales - 1:
                self.downsampling_modules.append(Downsample(in_ch))
                h_cs.append(in_ch)

        self.downsampling_modules = nn.ModuleList(self.downsampling_modules)

        # Middle
        self.middle_modules = []

        self.middle_modules.append(
            ResBlock(in_ch, in_ch, temb_dim=self.expanded_time_dim,
                dropout=dropout, skip_rescale=self.skip_rescale)
        )
        self.middle_modules.append(
            AttnBlock(in_ch, skip_rescale=self.skip_rescale)
        )
        self.middle_modules.append(
            ResBlock(in_ch, in_ch, temb_dim=self.expanded_time_dim,
                dropout=dropout, skip_rescale=self.skip_rescale)
        )
        self.middle_modules = nn.ModuleList(self.middle_modules)

        # Upsampling
        self.upsampling_modules = []

        for scale_count in reversed(range(self.num_scales)):
            for res_count in range(self.num_res_blocks+1):
                out_ch = self.ch * self.ch_mult[scale_count]
                self.upsampling_modules.append(
                    ResBlock(in_ch + h_cs.pop(), 
                        out_ch,
                        temb_dim=self.expanded_time_dim,
                        dropout=dropout,
                        skip_rescale=self.skip_rescale
                    )
                )
                in_ch = out_ch

                if scale_count == self.scale_count_to_put_attn:
                    self.upsampling_modules.append(
                        AttnBlock(in_ch, skip_rescale=self.skip_rescale)
                    )
            if scale_count != 0:
                self.upsampling_modules.append(Upsample(in_ch))

        self.upsampling_modules = nn.ModuleList(self.upsampling_modules)

        assert len(h_cs) == 0

        # output
        self.output_modules = []
        
        self.output_modules.append(
            nn.GroupNorm(min(in_ch//4, 32), in_ch, eps=1e-6)
        )

        self.output_modules.append(
            nn.Conv2d(in_ch, self.output_channels, kernel_size=3, padding=1)
        )
        self.output_modules = nn.ModuleList(self.output_modules)


    def _center_data(self, x):
        out = (x - self.data_min_max[0]) / (self.data_min_max[1] - self.data_min_max[0]) # [0, 1]
        return 2 * out - 1 # to put it in [-1, 1]

    def _time_embedding(self, timesteps):
        if self.do_time_embed:
            temb = transformer_timestep_embedding(
                timesteps * self.time_scale_factor, self.time_embed_dim
            )
            temb = self.temb_modules[0](temb)
            temb = self.temb_modules[1](self.act(temb))
        else:
            temb = None

        return temb

    def _do_input_conv(self, h):
        h = self.input_conv(h)
        hs = [h]
        return h, hs

    def _do_downsampling(self, h, hs, temb):
        m_idx = 0
        for scale_count in range(self.num_scales):
            for res_count in range(self.num_res_blocks):
                h = self.downsampling_modules[m_idx](h, temb)
                m_idx += 1
                if scale_count == self.scale_count_to_put_attn:
                    h = self.downsampling_modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            if scale_count != self.num_scales - 1:
                h = self.downsampling_modules[m_idx](h)
                hs.append(h)
                m_idx += 1

        assert m_idx == len(self.downsampling_modules)

        return h, hs

    def _do_middle(self, h, temb):
        m_idx = 0
        h = self.middle_modules[m_idx](h, temb)
        m_idx += 1
        h = self.middle_modules[m_idx](h)
        m_idx += 1
        h = self.middle_modules[m_idx](h, temb)
        m_idx += 1

        assert m_idx == len(self.middle_modules)

        return h

    def _do_upsampling(self, h, hs, temb):
        m_idx = 0
        for scale_count in reversed(range(self.num_scales)):
            for res_count in range(self.num_res_blocks+1):
                h = self.upsampling_modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

                if scale_count == self.scale_count_to_put_attn:
                    h = self.upsampling_modules[m_idx](h)
                    m_idx += 1

            if scale_count != 0:
                h = self.upsampling_modules[m_idx](h)
                m_idx += 1

        assert len(hs) == 0
        assert m_idx == len(self.upsampling_modules)

        return h

    def _do_output(self, h):

        h = self.output_modules[0](h)
        h = self.act(h)
        h = self.output_modules[1](h)

        return h

    def _logistic_output_res(self,
        h, #  ["B", "twoC", "H", "W"]
        centered_x_in, # ["B", "C", "H", "W"]
    ):
        B, twoC, H, W = h.shape
        C = twoC//2
        h[:, 0:C, :, :] = torch.tanh(centered_x_in + h[:, 0:C, :, :])
        return h

    def forward(self,
        x, # ["B", "C", "H", "W"]
        timesteps=None, # ["B"]
    ):

        h = self._center_data(x)
        centered_x_in = h

        temb = self._time_embedding(timesteps)

        h, hs = self._do_input_conv(h)

        h, hs = self._do_downsampling(h, hs, temb)

        h = self._do_middle(h, temb)

        h = self._do_upsampling(h, hs, temb)

        h = self._do_output(h)

        # h (B, 2*C, H, W)
        h = self._logistic_output_res(h, centered_x_in)

        return h


class UNetClassifier(nn.Module):
    """
    A classifier model p(y | z_t, t).
    The architecture primarily follows Dhariwal and Nichol 2021, 
    which uses the downsampling trunk of the UNet with an attention pool 
    at the 8x8 layer to produce the final output
    We use the same UNet as the one used for the denoising model
    and apply global average pooling at the 4x4 layer
    """
    def __init__(self, cfg):
        super().__init__()

        self.data_shape = cfg.data.shape
        ch = cfg.model.ch
        num_res_blocks = cfg.model.num_res_blocks
        num_scales = cfg.model.num_scales
        ch_mult = cfg.model.ch_mult
        input_channels = cfg.model.input_channels
        output_channels = cfg.model.num_classes
        scale_count_to_put_attn = cfg.model.scale_count_to_put_attn
        data_min_max = cfg.model.data_min_max
        dropout = cfg.model.dropout
        skip_rescale = cfg.model.skip_rescale
        do_time_embed = True
        time_scale_factor = cfg.model.time_scale_factor
        time_embed_dim = cfg.model.time_embed_dim

        assert num_scales == len(ch_mult)

        self.ch = ch
        self.num_res_blocks = num_res_blocks
        self.num_scales = num_scales
        self.ch_mult = ch_mult
        self.input_channels = input_channels
        # Number of output channels is the number of classes
        self.output_channels = output_channels 
        self.scale_count_to_put_attn = scale_count_to_put_attn
        self.data_min_max = data_min_max # tuple of min and max value of input so it can be rescaled to [-1, 1]
        self.dropout = dropout
        self.skip_rescale = skip_rescale
        self.do_time_embed = do_time_embed # Whether to add in time embeddings
        self.time_scale_factor = time_scale_factor # scale to make the range of times be 0 to 1000
        self.time_embed_dim = time_embed_dim

        self.act = nn.functional.silu

        if self.do_time_embed:
            self.temb_modules = []
            self.temb_modules.append(nn.Linear(self.time_embed_dim, self.time_embed_dim*4))
            nn.init.zeros_(self.temb_modules[-1].bias)
            self.temb_modules.append(nn.Linear(self.time_embed_dim*4, self.time_embed_dim*4))
            nn.init.zeros_(self.temb_modules[-1].bias)
            self.temb_modules = nn.ModuleList(self.temb_modules)

        self.expanded_time_dim = 4 * self.time_embed_dim if self.do_time_embed else None

        self.input_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=self.ch,
            kernel_size=3, padding=1
        )

        h_cs = [self.ch]
        in_ch = self.ch

        # Downsampling
        self.downsampling_modules = []

        for scale_count in range(self.num_scales):
            for res_count in range(self.num_res_blocks):
                out_ch = self.ch * self.ch_mult[scale_count]
                self.downsampling_modules.append(
                    ResBlock(in_ch, out_ch, temb_dim=self.expanded_time_dim,
                        dropout=dropout, skip_rescale=self.skip_rescale)
                )
                in_ch = out_ch
                h_cs.append(in_ch)
                if scale_count == self.scale_count_to_put_attn:
                    self.downsampling_modules.append(
                        AttnBlock(in_ch, skip_rescale=self.skip_rescale)
                    )

            if scale_count != self.num_scales - 1:
                self.downsampling_modules.append(Downsample(in_ch))
                h_cs.append(in_ch)

        self.downsampling_modules = nn.ModuleList(self.downsampling_modules)

        # Output layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc = nn.Linear(in_ch, self.output_channels)  # Fully connected layer for classification

    def _center_data(self, x):
        out = (x - self.data_min_max[0]) / (self.data_min_max[1] - self.data_min_max[0]) # [0, 1]
        return 2 * out - 1 # to put it in [-1, 1]

    def _time_embedding(self, timesteps):
        if self.do_time_embed:
            temb = transformer_timestep_embedding(
                timesteps * self.time_scale_factor, self.time_embed_dim
            )
            temb = self.temb_modules[0](temb)
            temb = self.temb_modules[1](self.act(temb))
        else:
            temb = None

        return temb

    def _do_input_conv(self, h):
        h = self.input_conv(h)
        hs = [h]
        return h, hs

    def _do_downsampling(self, h, hs, temb):
        m_idx = 0
        for scale_count in range(self.num_scales):
            for res_count in range(self.num_res_blocks):
                h = self.downsampling_modules[m_idx](h, temb)
                m_idx += 1
                if scale_count == self.scale_count_to_put_attn:
                    h = self.downsampling_modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            if scale_count != self.num_scales - 1:
                h = self.downsampling_modules[m_idx](h)
                hs.append(h)
                m_idx += 1

        assert m_idx == len(self.downsampling_modules)

        return h, hs

    def forward(self,
        x, # Shape (B, D)
        timesteps=None, # Shape (B,)
    ):
        B = x.shape[0]
        C, H, W = self.data_shape
        x = x.view(B, C, H, W)

        h = self._center_data(x)
        temb = self._time_embedding(timesteps)
        h, hs = self._do_input_conv(h)
        # h: shape (B, C=256, 8, 8)
        h, hs = self._do_downsampling(h, hs, temb)

        h = self.global_pool(h).view(B, h.shape[1])  # Shape (B, C=256)
        logits = self.fc(h)  # Shape (B, #classes)
        return logits

    def log_prob(self, batch_z_t, batch_t, batch_y, return_logits=False):
        """
        Evaluate the log probability of a batch of noisy inputs under the model
        against their class labels
        """
        logits = self.forward(batch_z_t, batch_t)
        # Per data point log probability of the true labels,
        # equivalent to the negative cross entropy
        # Shape (B,)
        log_prob = - F.cross_entropy(logits, batch_y, reduction='none')
        if not return_logits:
            return log_prob
        else:
            return log_prob, logits        


# From https://github.com/yang-song/score_sde_pytorch/ which is from
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  # emb = math.log(2.) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
  # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb