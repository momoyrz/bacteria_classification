import math
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import DeformConv2d
from functools import partial

from einops import rearrange, reduce
from timm.layers.weight_init import trunc_normal_
from timm.layers.activations import *
from timm.layers import DropPath
from HybridNet import HybridNetworkModule


# ========== For Common ==========
class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


class LayerNorm3d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b t h w c -> b c t h w').contiguous()
        return x


def get_conv(conv_layer='conv_2d'):
    conv_dict = {
        'conv_2d': nn.Conv2d,
        'conv_3d': nn.Conv3d,
        'dcn2_2d': DCN2,
        'conv_2ds': Conv2ds,
    }
    return conv_dict[conv_layer]


def get_norm(norm_layer='in_1d'):
    eps = 1e-6
    norm_dict = {
        'none': nn.Identity,
        'in_1d': partial(nn.InstanceNorm1d, eps=eps),
        'in_2d': partial(nn.InstanceNorm2d, eps=eps),
        'in_3d': partial(nn.InstanceNorm3d, eps=eps),
        'bn_1d': partial(nn.BatchNorm1d, eps=eps),
        'bn_2d': partial(nn.BatchNorm2d, eps=eps),
        'bn_3d': partial(nn.BatchNorm3d, eps=eps),
        'gn': partial(nn.GroupNorm, eps=eps),
        'ln_1d': partial(nn.LayerNorm, eps=eps),
        'ln_2d': partial(LayerNorm2d, eps=eps),
        'ln_3d': partial(LayerNorm3d, eps=eps),
        'bn_2ds': partial(BatchNorm2ds, eps=eps),
    }
    return norm_dict[norm_layer]


def get_act(act_layer='relu'):
    act_dict = {
        'none': nn.Identity,
        'sigmoid': Sigmoid,
        'swish': Swish,
        'mish': Mish,
        'hsigmoid': HardSigmoid,
        'hswish': HardSwish,
        'hmish': HardMish,
        'tanh': Tanh,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'prelu': PReLU,
        'gelu': GELU,
        'silu': nn.SiLU
    }
    return act_dict[act_layer]


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(1, 1, dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerScale2D(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerScale3D(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DCN2(nn.Module):
    # ref: https://github.com/WenmuZhou/DBNet.pytorch/blob/678b2ae55e018c6c16d5ac182558517a154a91ed/models/backbone/resnet.py
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 deform_groups=4):
        super().__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(dim_in, deform_groups * offset_channels, kernel_size=3, stride=stride, padding=1)
        self.conv = DeformConv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        offset = self.conv_offset(x)
        x = self.conv(x, offset)
        return x


class Conv2ds(nn.Conv2d):

    def __init__(self, dim_in, dim_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        super().__init__(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device,
                         dtype)
        self.in_channels_i = dim_in
        self.out_channels_i = dim_out
        self.groups_i = groups

    def forward(self, x, dim_in=None, dim_out=None):
        self.groups = dim_in if self.groups_i != 1 else self.groups_i
        in_channels = dim_in if dim_in else self.in_channels_i
        out_channels = dim_out if dim_out else self.out_channels_i
        weight = self.weight[:out_channels, :in_channels, :, :]
        bias = self.bias[:out_channels] if self.bias is not None else self.bias
        return self._conv_forward(x, weight, bias)


class BatchNorm2ds(nn.BatchNorm2d):

    def __init__(self, dim_in, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__(dim_in, eps, momentum, affine, track_running_stats, device, dtype)
        self.num_features = dim_in

    def forward(self, x, dim_in=None):
        self._check_input_dim(x)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        running_mean = self.running_mean[:dim_in]
        running_var = self.running_var[:dim_in]
        weight = self.weight[:dim_in]
        bias = self.bias[:dim_in]
        return F.batch_norm(x,
                            running_mean if not self.training or self.track_running_stats else None,
                            running_var if not self.training or self.track_running_stats else None,
                            weight, bias, bn_training, exponential_average_factor, self.eps,
                            )


class ConvNormAct(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False, padding_mode='zeros',
                 skip=False, conv_layer='conv_2d', norm_layer='bn_2d', act_layer='relu', inplace=True,
                 drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.has_skip = skip and dim_in == dim_out

        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = [math.ceil(((k - 1) * dilation + 1 - s) / 2) for k, s in zip(kernel_size, stride)]
        else:
            padding = math.ceil(((kernel_size - 1) * dilation + 1 - stride) / 2)
        if conv_layer in ['conv_2d', 'conv_2ds', 'conv_3d']:
            self.conv = get_conv(conv_layer)(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias,
                                             padding_mode=padding_mode)
        elif conv_layer in ['dcn2_2d', 'dcn2_2d_mmcv']:
            self.conv = get_conv(conv_layer)(dim_in, dim_out, kernel_size, stride, padding, dilation, groups,
                                             deform_groups=4, bias=bias)
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x, dim_in=None, dim_out=None):
        shortcut = x
        x = self.conv(x, dim_in=dim_in, dim_out=dim_out) if self.conv_layer in ['conv_2ds'] else self.conv(x)
        x = self.norm(x, dim_in=dim_out) if self.norm_layer in ['bn_2ds'] else self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


# ========== Multi-Scale Populations, for down-sampling and inductive bias ==========
class MSPatchEmb(nn.Module):

    def __init__(self, dim_in, emb_dim, kernel_size=2, c_group=-1, stride=1, dilations=[1, 2, 3],
                 norm_layer='bn_2d', act_layer='silu'):
        super().__init__()
        self.dilation_num = len(dilations)
        assert dim_in % c_group == 0
        c_group = math.gcd(dim_in, emb_dim) if c_group == -1 else c_group
        self.convs = nn.ModuleList()
        for i in range(len(dilations)):
            padding = math.ceil(((kernel_size - 1) * dilations[i] + 1 - stride) / 2)
            self.convs.append(nn.Sequential(
                nn.Conv2d(dim_in, emb_dim, kernel_size, stride, padding, dilations[i], groups=c_group),
                get_norm(norm_layer)(emb_dim),
                get_act(act_layer)(emb_dim)))

    def forward(self, x):
        if self.dilation_num == 1:
            x = self.convs[0](x)
        else:
            x = torch.cat([self.convs[i](x).unsqueeze(dim=-1) for i in range(self.dilation_num)], dim=-1)
            x = reduce(x, 'b c h w n -> b c h w', 'mean').contiguous()
        return x


def gen_cfg(opss=['d2.0', 'd3.0', 's1.0d3.0', 's1.0d3.0'], depths=[3, 3, 9, 3]):
    cfg = []
    for ops, depth in zip(opss, depths):
        ops = re.findall(r'[a-zA-Z]\d\.\d*', ops)
        ops = {op[0]: float(op[1:]) for op in ops}
        for i in range(depth):
            if i == 0:
                cfg += ['d{:.3f}'.format(sum(list(ops.values())) * 2)]
            else:
                cfg_l = ''
                for k, v in ops.items():
                    cfg_l += '{}{:.3f}'.format(k, v)
                cfg += [cfg_l]
    return cfg


# ======== MobileMamba Implementation Without MRFFI ========

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module('dwconv3x3',
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels,
                                  bias=False))
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))

        # Initialize batch norm weights
        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # Fuse dwconv3x3 and bn1
        dwconv3x3, bn1, relu, dwconv1x1, bn2 = self._modules.values()

        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3.weight * w1[:, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5

        fused_dwconv3x3 = nn.Conv2d(w1.size(1) * dwconv3x3.groups, w1.size(0), w1.shape[2:], stride=dwconv3x3.stride,
                                    padding=dwconv3x3.padding, dilation=dwconv3x3.dilation, groups=dwconv3x3.groups,
                                    device=dwconv3x3.weight.device)
        fused_dwconv3x3.weight.data.copy_(w1)
        fused_dwconv3x3.bias.data.copy_(b1)

        # Fuse dwconv1x1 and bn2
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5

        fused_dwconv1x1 = nn.Conv2d(w2.size(1) * dwconv1x1.groups, w2.size(0), w2.shape[2:], stride=dwconv1x1.stride,
                                    padding=dwconv1x1.padding, dilation=dwconv1x1.dilation, groups=dwconv1x1.groups,
                                    device=dwconv1x1.weight.device)
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)

        # Create a new sequential model with fused layers
        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, )
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim,)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0,)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


def nearest_multiple_of_16(n):
    if n % 16 == 0:
        return n
    else:
        lower_multiple = (n // 16) * 16
        upper_multiple = lower_multiple + 16

        if (n - lower_multiple) < (upper_multiple - n):
            return lower_multiple
        else:
            return upper_multiple

class MobileMambaBlockWindow(torch.nn.Module):
    def __init__(self, dim, kernels=3):
        super().__init__()
        self.dim = dim
        # 使用HybridNetwork直接替代SimplifiedModule
        self.attn = HybridNetworkModule.HybridNetwork(
            in_channels=dim,
            hidden_channels=dim*2,  # 扩展通道
            out_channels=dim,
            mamba_d_state=16,
            mamba_d_conv=3,
            mamba_expand_ratio=2.0,
            awt_kernel_size=kernels,
            awt_wt_levels=1,
            awt_wt_type='db1',
            dropout=0.0
        )

    def forward(self, x):
        x = self.attn(x)
        return x


class MobileMambaBlock(torch.nn.Module):
    def __init__(self, type, ed, kernels=3, drop_path=0., has_skip=True):
        super().__init__()

        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))

        if type == 's':
            self.mixer = Residual(MobileMambaBlockWindow(ed, kernels=kernels))
        else:
            # Fallback for other types if needed
            self.mixer = nn.Identity()

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.,))
        self.ffn1 = Residual(FFN(ed, int(ed * 2)))

        self.has_skip = has_skip
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class MobileMamba(torch.nn.Module):
    def __init__(self, img_size=192,
                 in_chans=3,
                 num_classes=1000,
                 stages=['s', 's', 's'],
                 embed_dim=[144, 272, 368],
                 depth=[1, 2, 2],
                 kernels=[7, 5, 3],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=False, drop_path=0.):
        super().__init__()

        resolution = img_size
        # Patch embedding
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1),
                                               torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1,
                                                         ), torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1,
                                                         ), torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1,
                                                         ))

        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depth))]

        # Build MobileMamba blocks
        for i, (stg, ed, dpth, ks, do) in enumerate(
                zip(stages, embed_dim, depth, kernels, down_ops)):
            dpr = dprs[sum(depth[:i]):sum(depth[:i + 1])]
            for d in range(dpth):
                eval('self.blocks' + str(i + 1)).append(MobileMambaBlock(stg, ed, kernels=ks, drop_path=dpr[d]))
            if do[0] == 'subsample':
                # Build MobileMamba downsample block
                # ('Subsample' stride)
                blk = eval('self.blocks' + str(i + 2))
                blk.append(torch.nn.Sequential(Residual(
                    Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i])),
                                               Residual(FFN(embed_dim[i], int(embed_dim[i] * 2))), ))
                blk.append(PatchMerging(*embed_dim[i:i + 2]))
                blk.append(torch.nn.Sequential(Residual(
                    Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1],)),
                                               Residual(
                                                   FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2))), ))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)

        # Classification head
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x

def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


# T2 configuration only
def MobileMamba_T2(num_classes=, pretrained=False, distillation=False, fuse=False):
    model_cfg = {
        'img_size': 192,
        'embed_dim': [144, 272, 368],
        'depth': [1, 2, 2],
        'kernels': [7, 5, 3],
        'stages': ['s', 's', 's'],
        'drop_path': 0,
        'down_ops': [['subsample', 2], ['subsample', 2], ['']]
    }
    model = MobileMamba(num_classes=num_classes, distillation=distillation, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model