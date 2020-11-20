import yaml
import math
from easydict import EasyDict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from template_lib.d2.layers_v2.build import D2LAYERv2_REGISTRY, build_d2layer_v2
from template_lib.utils import get_attr_kwargs
from template_lib.v2.config import update_config


@D2LAYERv2_REGISTRY.register()
class ClassModulatedConv2d(nn.Module):
  def __init__(self, cfg, **kwargs):
    super().__init__()

    # fmt: off
    self.n_classes                 = get_attr_kwargs(cfg, 'n_classes', **kwargs)
    self.style_dim                 = get_attr_kwargs(cfg, 'style_dim', default=None, **kwargs)
    self.in_channels               = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels              = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.kernel_size               = get_attr_kwargs(cfg, 'kernel_size', **kwargs)
    self.zero_embedding            = get_attr_kwargs(cfg, 'zero_embedding', default=False, **kwargs)
    self.spectral_norm             = get_attr_kwargs(cfg, 'spectral_norm', default=False, **kwargs)
    self.use_affine                = get_attr_kwargs(cfg, 'use_affine', default=False, **kwargs)
    self.demodulate                = get_attr_kwargs(cfg, 'demodulate', default=True, **kwargs)
    self.init                      = get_attr_kwargs(cfg, 'init', default='ortho', **kwargs)
    self.style_plus_one            = get_attr_kwargs(cfg, 'style_plus_one', default=True, **kwargs)
    # fmt: on

    self.embedding = nn.Embedding(self.n_classes, self.style_dim)
    if self.zero_embedding:
      init.zeros_(self.embedding.weight)

    if self.spectral_norm:
      self.embedding = nn.utils.spectral_norm(self.embedding)

    fan_in = self.in_channels * self.kernel_size ** 2
    self.scale = 1 / math.sqrt(fan_in)
    self.padding = self.kernel_size // 2

    self.weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
    if self.init == 'ortho':
      init.orthogonal_(self.weight)
    elif self.init == 'normal':
      init.normal_(self.weight, 0, 1.)
    elif self.init in ['glorot', 'xavier']:
      init.xavier_uniform_(self.weight)
    else:
      print('Init style not recognized...')

    if self.use_affine:
      self.modulation = nn.Linear(self.style_dim, self.in_channels)
      # self.modulation = EqualLinear(self.style_dim, self.in_channels, bias_init=1)
    pass

  def forward(self, input):
    batch, in_channel, height, width = input.shape
    style = self.embedding(torch.arange(self.n_classes).to(input.device))
    if self.use_affine:
      style = self.modulation(style).view(self.out_channels, in_channel, 1, 1)
    else:
      style = style.view(self.out_channels, in_channel, 1, 1)

    if self.style_plus_one:
      weight = self.scale * self.weight * (style + 1)
    else:
      weight = self.scale * self.weight * style

    if self.demodulate:
      demod = torch.rsqrt(weight.pow(2).sum([1, 2, 3]) + 1e-8)
      weight = weight * demod.view(self.out_channels, 1, 1, 1)

    out = F.conv2d(input, weight, padding=self.padding)

    return out

  def __repr__(self):
    return (f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, {self.kernel_size}), '
            f'use_affine={self.use_affine}, style_plus_one={self.style_plus_one}, '
            f'demodulate={self.demodulate}')

  @staticmethod
  def test_case():
    import template_lib.d2.layers_v2.convs

    # test use_affine=false
    cfg_str = """
              name: "ClassModulatedConv2d"
              update_cfg: true
              n_classes: 10
              in_channels: 128
              out_channels: 10
              use_affine: false
              style_dim: 128
              """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = ClassModulatedConv2d.update_cfg(cfg)

    op = build_d2layer_v2(cfg)
    op.cuda()

    bs = 2
    x = torch.randn(bs, op.in_channels, 8, 8).cuda()
    out = op(x)

  @staticmethod
  def update_cfg(cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
      name: "ClassModulatedConv2d"
      in_channels: 128
      out_channels: 128
      kernel_size: 3      
      use_affine: true
      style_dim: 256      
      demodulate: true
      """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg


if __name__ == '__main__':
  ClassModulatedConv2d.test_case()
