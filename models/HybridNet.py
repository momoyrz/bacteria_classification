import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import repeat, rearrange
from typing import Optional, List, Tuple, Union
from functools import partial

class AWTConvModule:
    @staticmethod
    def wavelet_transform(x, h_analysis):
        """小波变换函数"""
        B, C, H, W = x.shape
        h = h_analysis
        x_ll = F.conv2d(x, h[0:1, ...], stride=2, padding=1, groups=C)
        x_lh = F.conv2d(x, h[1:2, ...], stride=2, padding=1, groups=C)
        x_hl = F.conv2d(x, h[2:3, ...], stride=2, padding=1, groups=C)
        x_hh = F.conv2d(x, h[3:4, ...], stride=2, padding=1, groups=C)

        x = torch.stack([x_ll, x_lh, x_hl, x_hh], dim=2)
        return x

    @staticmethod
    def inverse_wavelet_transform(x, h_synthesis):
        """小波逆变换函数"""
        B, C, _, H, W = x.shape
        h = h_synthesis

        x_ll = x[:, :, 0, :, :]
        x_lh = x[:, :, 1, :, :]
        x_hl = x[:, :, 2, :, :]
        x_hh = x[:, :, 3, :, :]

        x = (F.conv_transpose2d(x_ll, h[0:1, ...], stride=2, padding=1, groups=C) +
             F.conv_transpose2d(x_lh, h[1:2, ...], stride=2, padding=1, groups=C) +
             F.conv_transpose2d(x_hl, h[2:3, ...], stride=2, padding=1, groups=C) +
             F.conv_transpose2d(x_hh, h[3:4, ...], stride=2, padding=1, groups=C))

        return x

    @staticmethod
    def create_wavelet_filter(wave, in_channels, out_channels, dtype):
        """创建小波滤波器"""
        import pywt
        wave_length = 0
        if isinstance(wave, int):
            wave_length = wave
            wave = 'db1'
        wavelet = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(wavelet.dec_hi)
        dec_lo = torch.Tensor(wavelet.dec_lo)

        rec_hi = torch.Tensor(wavelet.rec_hi)
        rec_lo = torch.Tensor(wavelet.rec_lo)

        filters = torch.zeros(4, 1, dec_lo.shape[0], dec_lo.shape[0])
        wt_filters = torch.zeros(4, 1, dec_lo.shape[0], dec_lo.shape[0])
        iwt_filters = torch.zeros(4, 1, rec_lo.shape[0], rec_lo.shape[0])

        for i in range(dec_lo.shape[0]):
            for j in range(dec_lo.shape[0]):
                filters[0, 0, i, j] = dec_lo[i] * dec_lo[j]
                filters[1, 0, i, j] = dec_lo[i] * dec_hi[j]
                filters[2, 0, i, j] = dec_hi[i] * dec_lo[j]
                filters[3, 0, i, j] = dec_hi[i] * dec_hi[j]

        wt_filters = filters.clone()

        for i in range(rec_lo.shape[0]):
            for j in range(rec_lo.shape[0]):
                iwt_filters[0, 0, i, j] = rec_lo[i] * rec_lo[j]
                iwt_filters[1, 0, i, j] = rec_lo[i] * rec_hi[j]
                iwt_filters[2, 0, i, j] = rec_hi[i] * rec_lo[j]
                iwt_filters[3, 0, i, j] = rec_hi[i] * rec_hi[j]

        ifm_weight = iwt_filters.repeat(out_channels, in_channels, 1, 1)
        fm_weight = wt_filters.repeat(out_channels, in_channels, 1, 1)

        return ifm_weight.to(dtype), fm_weight.to(dtype)

    class ARConv(nn.Module):
        """自适应矩形卷积模块"""

        def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, l_max=9, w_max=9, flag=False,
                     modulation=True):
            super(AWTConvModule.ARConv, self).__init__()
            self.lmax = l_max
            self.wmax = w_max
            self.inc = inc
            self.outc = outc
            self.kernel_size = kernel_size
            self.padding = padding
            self.stride = stride
            self.zero_padding = nn.ZeroPad2d(padding)
            self.flag = flag
            self.modulation = modulation
            self.i_list = [33, 35, 53, 37, 73, 55, 57, 75, 77]
            self.convs = nn.ModuleList(
                [
                    nn.Conv2d(inc, outc, kernel_size=(i // 10, i % 10), stride=(i // 10, i % 10), padding=0)
                    for i in self.i_list
                ]
            )
            self.m_conv = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
                nn.LeakyReLU(),
                nn.Dropout2d(0.3),
                nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
                nn.LeakyReLU(),
                nn.Dropout2d(0.3),
                nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
                nn.Tanh()
            )
            self.b_conv = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
                nn.LeakyReLU(),
                nn.Dropout2d(0.3),
                nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
                nn.LeakyReLU(),
                nn.Dropout2d(0.3),
                nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride)
            )
            self.p_conv = nn.Sequential(
                nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm2d(inc),
                nn.LeakyReLU(),
                nn.Dropout2d(0),
                nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm2d(inc),
                nn.LeakyReLU(),
            )
            self.l_conv = nn.Sequential(
                nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm2d(1),
                nn.LeakyReLU(),
                nn.Dropout2d(0),
                nn.Conv2d(1, 1, 1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            self.w_conv = nn.Sequential(
                nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm2d(1),
                nn.LeakyReLU(),
                nn.Dropout2d(0),
                nn.Conv2d(1, 1, 1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            self.dropout1 = nn.Dropout(0.3)
            self.dropout2 = nn.Dropout2d(0.3)
            self.hook_handles = []
            self.hook_handles.append(self.m_conv[0].register_full_backward_hook(self._set_lr))
            self.hook_handles.append(self.m_conv[1].register_full_backward_hook(self._set_lr))
            self.hook_handles.append(self.b_conv[0].register_full_backward_hook(self._set_lr))
            self.hook_handles.append(self.b_conv[1].register_full_backward_hook(self._set_lr))
            self.hook_handles.append(self.p_conv[0].register_full_backward_hook(self._set_lr))
            self.hook_handles.append(self.p_conv[1].register_full_backward_hook(self._set_lr))
            self.hook_handles.append(self.l_conv[0].register_full_backward_hook(self._set_lr))
            self.hook_handles.append(self.l_conv[1].register_full_backward_hook(self._set_lr))
            self.hook_handles.append(self.w_conv[0].register_full_backward_hook(self._set_lr))
            self.hook_handles.append(self.w_conv[1].register_full_backward_hook(self._set_lr))

            self.reserved_NXY = nn.Parameter(torch.tensor([3, 3], dtype=torch.int32), requires_grad=False)

        @staticmethod
        def _set_lr(module, grad_input, grad_output):
            grad_input = tuple(g * 0.1 if g is not None else None for g in grad_input)
            grad_output = tuple(g * 0.1 if g is not None else None for g in grad_output)
            return grad_input

        def remove_hooks(self):
            for handle in self.hook_handles:
                handle.remove()  # 移除钩子函数
            self.hook_handles.clear()  # 清空句柄列表

        def forward(self, x, epoch, hw_range):
            assert isinstance(hw_range, list) and len(
                hw_range) == 2, "hw_range should be a list with 2 elements, represent the range of h w"
            scale = hw_range[1] // 9
            if hw_range[0] == 1 and hw_range[1] == 3:
                scale = 1
            m = self.m_conv(x)
            bias = self.b_conv(x)
            offset = self.p_conv(x * 100)
            l = self.l_conv(offset) * (hw_range[1] - 1) + 1  # b, 1, h, w
            w = self.w_conv(offset) * (hw_range[1] - 1) + 1  # b, 1, h, w
            if epoch <= 100:
                mean_l = l.mean(dim=0).mean(dim=1).mean(dim=1)
                mean_w = w.mean(dim=0).mean(dim=1).mean(dim=1)
                N_X = int(mean_l // scale)
                N_Y = int(mean_w // scale)

                def phi(x):
                    if x % 2 == 0:
                        x -= 1
                    return x

                N_X, N_Y = phi(N_X), phi(N_Y)
                N_X, N_Y = max(N_X, 3), max(N_Y, 3)
                N_X, N_Y = min(N_X, 7), min(N_Y, 7)
                if epoch == 100:
                    self.reserved_NXY = nn.Parameter(
                        torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device),
                        requires_grad=False
                    )
            else:
                N_X = self.reserved_NXY[0]
                N_Y = self.reserved_NXY[1]

            N = N_X * N_Y
            # print(N_X, N_Y)
            l = l.repeat([1, N, 1, 1])
            w = w.repeat([1, N, 1, 1])
            offset = torch.cat((l, w), dim=1)
            dtype = offset.data.type()
            if self.padding:
                x = self.zero_padding(x)
            p = self._get_p(offset, dtype, N_X, N_Y)  # (b, 2*N, h, w)
            p = p.contiguous().permute(0, 2, 3, 1)  # (b, h, w, 2*N)
            q_lt = p.detach().floor()
            q_rb = q_lt + 1
            q_lt = torch.cat(
                [
                    torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                    torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
                ],
                dim=-1,
            ).long()
            q_rb = torch.cat(
                [
                    torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                    torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
                ],
                dim=-1,
            ).long()
            q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
            q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
            # clip p
            p = torch.cat(
                [
                    torch.clamp(p[..., :N], 0, x.size(2) - 1),
                    torch.clamp(p[..., N:], 0, x.size(3) - 1),
                ],
                dim=-1,
            )
            # bilinear kernel (b, h, w, N)
            g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_lt[..., N:].type_as(p) - p[..., N:])
            )
            g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_rb[..., N:].type_as(p) - p[..., N:])
            )
            g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_lb[..., N:].type_as(p) - p[..., N:])
            )
            g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_rt[..., N:].type_as(p) - p[..., N:])
            )
            # (b, c, h, w, N)
            x_q_lt = self._get_x_q(x, q_lt, N)
            x_q_rb = self._get_x_q(x, q_rb, N)
            x_q_lb = self._get_x_q(x, q_lb, N)
            x_q_rt = self._get_x_q(x, q_rt, N)
            # (b, c, h, w, N)
            x_offset = (
                    g_lt.unsqueeze(dim=1) * x_q_lt
                    + g_rb.unsqueeze(dim=1) * x_q_rb
                    + g_lb.unsqueeze(dim=1) * x_q_lb
                    + g_rt.unsqueeze(dim=1) * x_q_rt
            )
            x_offset = self._reshape_x_offset(x_offset, N_X, N_Y)
            x_offset = self.dropout2(x_offset)
            x_offset = self.convs[self.i_list.index(N_X * 10 + N_Y)](x_offset)
            out = x_offset * m + bias
            return out

        def _get_p_n(self, N, dtype, n_x, n_y):
            p_n_x, p_n_y = torch.meshgrid(
                torch.arange(-(n_x - 1) // 2, (n_x - 1) // 2 + 1),
                torch.arange(-(n_y - 1) // 2, (n_y - 1) // 2 + 1),
            )
            p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
            p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
            return p_n

        def _get_p_0(self, h, w, N, dtype):
            p_0_x, p_0_y = torch.meshgrid(
                torch.arange(1, h * self.stride + 1, self.stride),
                torch.arange(1, w * self.stride + 1, self.stride),
            )
            p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
            p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
            p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
            return p_0

        def _get_p(self, offset, dtype, n_x, n_y):
            N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
            L, W = offset.split([N, N], dim=1)
            L = L / n_x
            W = W / n_y
            offsett = torch.cat([L, W], dim=1)
            p_n = self._get_p_n(N, dtype, n_x, n_y)
            p_n = p_n.repeat([1, 1, h, w])
            p_0 = self._get_p_0(h, w, N, dtype)
            p = p_0 + offsett * p_n
            return p

        def _get_x_q(self, x, q, N):
            b, h, w, _ = q.size()
            padded_w = x.size(3)
            c = x.size(1)
            x = x.contiguous().view(b, c, -1)
            index = q[..., :N] * padded_w + q[..., N:]
            index = (
                index.contiguous()
                    .unsqueeze(dim=1)
                    .expand(-1, c, -1, -1, -1)
                    .contiguous()
                    .view(b, c, -1)
            )
            x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
            return x_offset

        @staticmethod
        def _reshape_x_offset(x_offset, n_x, n_y):
            b, c, h, w, N = x_offset.size()
            x_offset = torch.cat(
                [x_offset[..., s:s + n_y].contiguous().view(b, c, h, w * n_y) for s in range(0, N, n_y)],
                dim=-1)
            x_offset = x_offset.contiguous().view(b, c, h * n_x, w * n_y)
            return x_offset

    class _ScaleModule(nn.Module):

        def __init__(self, dims, init_scale=1.0, init_bias=0):
            super(AWTConvModule._ScaleModule, self).__init__()
            self.dims = dims
            self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
            self.bias = None

        def forward(self, x):
            return torch.mul(self.weight, x)

    class WTConv2d(nn.Module):
        """小波变换卷积模块，对四个子带分别应用ARConv"""

        def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
            super(AWTConvModule.WTConv2d, self).__init__()
            assert in_channels == out_channels
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.wt_levels = wt_levels
            self.stride = stride
            self.dilation = 1
            self.kernel_size = kernel_size

            # 小波变换滤波器
            self.wt_filter, self.iwt_filter = AWTConvModule.create_wavelet_filter(wt_type, in_channels, in_channels,
                                                                                  torch.float)
            self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
            self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

            # 为每个小波子带创建单独的ARConv
            for i in range(self.wt_levels):
                # 为LL子带创建ARConv
                setattr(self, f'll_conv_{i}', AWTConvModule.ARConv(
                    in_channels, in_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=1
                ))

                # 为LH子带创建ARConv
                setattr(self, f'lh_conv_{i}', AWTConvModule.ARConv(
                    in_channels, in_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=1
                ))

                # 为HL子带创建ARConv
                setattr(self, f'hl_conv_{i}', AWTConvModule.ARConv(
                    in_channels, in_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=1
                ))

                # 为HH子带创建ARConv
                setattr(self, f'hh_conv_{i}', AWTConvModule.ARConv(
                    in_channels, in_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=1
                ))

                # 为每个子带创建缩放模块
                setattr(self, f'll_scale_{i}', AWTConvModule._ScaleModule([1, in_channels, 1, 1], init_scale=0.1))
                setattr(self, f'lh_scale_{i}', AWTConvModule._ScaleModule([1, in_channels, 1, 1], init_scale=0.1))
                setattr(self, f'hl_scale_{i}', AWTConvModule._ScaleModule([1, in_channels, 1, 1], init_scale=0.1))
                setattr(self, f'hh_scale_{i}', AWTConvModule._ScaleModule([1, in_channels, 1, 1], init_scale=0.1))

            # 直接路径的ARConv
            self.direct_conv = AWTConvModule.ARConv(
                in_channels, in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1
            )
            self.direct_scale = AWTConvModule._ScaleModule([1, in_channels, 1, 1])

            # 如果需要下采样
            if self.stride > 1:
                self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
            else:
                self.do_stride = None

        def forward(self, x):
            # ARConv的参数
            epoch = 101  # 使用预设矩形核尺寸
            hw_range = [1, 9]  # 默认值

            # 直接路径处理
            direct_path = self.direct_scale(self.direct_conv(x, epoch, hw_range))

            # 小波变换路径处理
            x_ll_in_levels = []
            x_h_in_levels = []
            shapes_in_levels = []
            curr_x_ll = x

            # 小波变换前向传播，对每个子带分别应用ARConv
            for i in range(self.wt_levels):
                curr_shape = curr_x_ll.shape
                shapes_in_levels.append(curr_shape)

                # 处理形状不是偶数的情况
                if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                    curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                    curr_x_ll = F.pad(curr_x_ll, curr_pads)

                # 小波变换分解
                curr_x = AWTConvModule.wavelet_transform(curr_x_ll, self.wt_filter)

                # 获取各个子带
                ll_band = curr_x[:, :, 0, :, :]
                lh_band = curr_x[:, :, 1, :, :]
                hl_band = curr_x[:, :, 2, :, :]
                hh_band = curr_x[:, :, 3, :, :]

                # 对每个子带分别应用ARConv和缩放
                ll_conv = getattr(self, f'll_conv_{i}')
                lh_conv = getattr(self, f'lh_conv_{i}')
                hl_conv = getattr(self, f'hl_conv_{i}')
                hh_conv = getattr(self, f'hh_conv_{i}')

                ll_scale = getattr(self, f'll_scale_{i}')
                lh_scale = getattr(self, f'lh_scale_{i}')
                hl_scale = getattr(self, f'hl_scale_{i}')
                hh_scale = getattr(self, f'hh_scale_{i}')

                # 分别处理每个子带
                processed_ll = ll_scale(ll_conv(ll_band, epoch, hw_range))
                processed_lh = lh_scale(lh_conv(lh_band, epoch, hw_range))
                processed_hl = hl_scale(hl_conv(hl_band, epoch, hw_range))
                processed_hh = hh_scale(hh_conv(hh_band, epoch, hw_range))

                # 存储处理后的结果
                x_ll_in_levels.append(processed_ll)
                x_h_in_levels.append(torch.stack([processed_lh, processed_hl, processed_hh], dim=2))

                # 准备下一层的输入
                curr_x_ll = processed_ll

            # 小波逆变换
            next_x_ll = 0
            for i in range(self.wt_levels - 1, -1, -1):
                curr_x_ll = x_ll_in_levels.pop()
                curr_x_h = x_h_in_levels.pop()
                curr_shape = shapes_in_levels.pop()

                curr_x_ll = curr_x_ll + next_x_ll
                curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
                next_x_ll = AWTConvModule.inverse_wavelet_transform(curr_x, self.iwt_filter)
                next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

            wavelet_path = next_x_ll
            assert len(x_ll_in_levels) == 0

            # 合并两个路径
            output = wavelet_path + direct_path

            if self.do_stride is not None:
                output = self.do_stride(output)

            return output


class MambaModule:
    """
    Mamba (CustomScanMS2D) 模块的封装类，包含所有Mamba相关组件
    """

    # 检查SelectiveScan导入
    try:
        # 第一种方式：通过sscore模块导入
        from mamba_ssm.ops.selective_scan_interface import SelectiveScan
        SSMODE = "sscore"
    except ImportError:
        try:
            # 第二种方式：直接导入selective_scan_cuda
            import selective_scan_cuda
            from mamba_ssm.ops.selective_scan_interface import SelectiveScan
            SSMODE = "mamba_ssm"
        except ImportError:
            # 如果都导入失败，给出错误提示
            raise ImportError(
                "无法导入SelectiveScan。请确保已正确安装mamba_ssm或selective_scan_cuda。"
            )

    class BottConv(nn.Module):
        """瓶颈卷积模块"""

        def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
            super(MambaModule.BottConv, self).__init__()
            self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
            self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels,
                                       bias=False)
            self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

        def forward(self, x):
            x = self.pointwise_1(x)
            x = self.depthwise(x)
            x = self.pointwise_2(x)
            return x

    @staticmethod
    def get_norm_layer(norm_type, channels, num_groups):
        """获取归一化层"""
        if norm_type == 'GN':
            return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        else:
            # 修改为2D实例归一化层，与输入数据维度匹配
            return nn.InstanceNorm2d(channels)

    class GBC(nn.Module):
        """GBC模块"""

        def __init__(self, in_channels, norm_type='GN', stride=1):
            super(MambaModule.GBC, self).__init__()
            # 添加stride参数，允许下采样
            self.stride = stride
            self.block1 = nn.Sequential(
                MambaModule.BottConv(in_channels, in_channels, in_channels // 8, 3,
                                     stride if stride > 1 and self.stride == 1 else 1,
                                     1),
                MambaModule.get_norm_layer(norm_type, in_channels, in_channels // 16),
                nn.ReLU()
            )
            self.block2 = nn.Sequential(
                MambaModule.BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
                MambaModule.get_norm_layer(norm_type, in_channels, in_channels // 16),
                nn.ReLU()
            )
            self.block3 = nn.Sequential(
                MambaModule.BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
                MambaModule.get_norm_layer(norm_type, in_channels, in_channels // 16),
                nn.ReLU()
            )
            self.block4 = nn.Sequential(
                MambaModule.BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
                MambaModule.get_norm_layer(norm_type, in_channels, in_channels // 16),
                nn.ReLU()
            )

        def forward(self, x):
            residual = x
            if self.stride > 1:
                # 如果需要下采样，直接下采样残差连接
                residual = F.avg_pool2d(residual, self.stride)

            x1 = self.block1(x)
            x1 = self.block2(x1)
            x2 = self.block3(x if self.stride == 1 else F.avg_pool2d(x, self.stride))
            x = x1 * x2
            x = self.block4(x)
            return x + residual

    class CustomScanMS2D(nn.Module):
        """CustomScanMS2D模块，Mamba的2D实现"""

        def __init__(
                self,
                d_model=96,
                d_state=16,
                ssm_ratio=2.0,
                ssm_rank_ratio=2.0,
                dt_rank="auto",
                act_layer=nn.SiLU,
                d_conv=3,
                conv_bias=True,
                dropout=0.0,
                bias=False,
                dt_min=0.001,
                dt_max=0.1,
                dt_init="random",
                dt_scale=1.0,
                dt_init_floor=1e-4,
                use_checkpoint=False,
                **kwargs,
        ):
            super().__init__()
            factory_kwargs = {"device": None, "dtype": None}

            # Define dimensions
            d_expand = int(ssm_ratio * d_model)
            d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
            self.d_inner = d_inner
            self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
            self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
            self.d_conv = d_conv

            # Branch ratio
            b1_ratio = kwargs.get('b1_ratio', 0.5)
            b0_dim, b1_dim = int(d_inner * (1 - b1_ratio)), int(d_inner * b1_ratio)
            self.b0_dim = b0_dim
            self.b1_dim = b1_dim

            # Define normalization layers
            self.out_norm_b0 = nn.LayerNorm(b0_dim)
            self.out_norm_b1 = nn.LayerNorm(b1_dim)

            # Define branches
            self.K_b0 = 1  # branch 0, full resolution (1 scan direction)
            self.K_b1 = 3  # branch 1, downsampled resolution (3 scan directions)

            # Input projection (不再需要2倍尺寸，因为删除了门控)
            self.in_proj = nn.Linear(d_model, d_expand, bias=bias, **factory_kwargs)
            self.act = act_layer()

            # 使用GBC模块替代原来的卷积
            # 全分辨率分支
            self.gbc_b0 = MambaModule.GBC(b0_dim, norm_type='GN', stride=1)

            # 下采样分支，步长为2
            self.gbc_b1 = MambaModule.GBC(b1_dim, norm_type='GN', stride=2)

            # Low-rank projection (optional)
            self.ssm_low_rank = False
            if d_inner < d_expand:
                self.ssm_low_rank = True
                self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
                self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

            # X projection for both branches
            self.x_proj_b0 = [
                nn.Linear(b0_dim, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
                for _ in range(self.K_b0)
            ]
            self.x_proj_weight_b0 = nn.Parameter(torch.stack([t.weight for t in self.x_proj_b0], dim=0))
            del self.x_proj_b0

            self.x_proj_b1 = [
                nn.Linear(b1_dim, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
                for _ in range(self.K_b1)
            ]
            self.x_proj_weight_b1 = nn.Parameter(torch.stack([t.weight for t in self.x_proj_b1], dim=0))
            del self.x_proj_b1

            # DT projections for both branches
            self.dt_projs_b0 = [
                self.dt_init(self.dt_rank, b0_dim, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(self.K_b0)
            ]
            self.dt_projs_weight_b0 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs_b0], dim=0))
            self.dt_projs_bias_b0 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_b0], dim=0))
            del self.dt_projs_b0

            self.dt_projs_b1 = [
                self.dt_init(self.dt_rank, b1_dim, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(self.K_b1)
            ]
            self.dt_projs_weight_b1 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs_b1], dim=0))
            self.dt_projs_bias_b1 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_b1], dim=0))
            del self.dt_projs_b1

            # A and D parameters for both branches
            self.A_logs_b0 = self.A_log_init(self.d_state, b0_dim, copies=self.K_b0, merge=True)
            self.Ds_b0 = self.D_init(b0_dim, copies=self.K_b0, merge=True)

            self.A_logs_b1 = self.A_log_init(self.d_state, b1_dim, copies=self.K_b1, merge=True)
            self.Ds_b1 = self.D_init(b1_dim, copies=self.K_b1, merge=True)

            # Output projection
            self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
            self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

            # 删除了可选的Squeeze-and-Excitation模块

            self.use_checkpoint = use_checkpoint
            self.kwargs = kwargs

        # Helper methods for initialization
        @staticmethod
        def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                    **factory_kwargs):
            dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = dt_rank ** -0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)

            return dt_proj

        @staticmethod
        def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)  # Keep A_log in fp32
            if copies > 0:
                A_log = repeat(A_log, "d n -> r d n", r=copies)
                if merge:
                    A_log = A_log.flatten(0, 1)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
            return A_log

        @staticmethod
        def D_init(d_inner, copies=-1, device=None, merge=True):
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            if copies > 0:
                D = repeat(D, "n1 -> r n1", r=copies)
                if merge:
                    D = D.flatten(0, 1)
            D = nn.Parameter(D)  # Keep in fp32
            D._no_weight_decay = True
            return D

        # Custom scanning functions
        def z_scan(self, x):
            """
            Z-pattern scanning: Left to right for each row
            x: tensor of shape [B, C, H, W]
            returns: tensor of shape [B, C, H*W]
            """
            B, C, H, W = x.shape
            return x.view(B, C, H * W)

        def s_scan(self, x):
            """
            S-pattern (snake) scanning: alternate right-to-left and left-to-right
            x: tensor of shape [B, C, H, W]
            returns: tensor of shape [B, C, H*W]
            """
            B, C, H, W = x.shape
            rows = []
            for i in range(H):
                if i % 2 == 0:  # Even rows - right to left
                    rows.append(x[:, :, i, :].flip(dims=[-1]))
                else:  # Odd rows - left to right
                    rows.append(x[:, :, i, :])
            return torch.cat([r.view(B, C, -1) for r in rows], dim=-1)

        def alt_column_scan(self, x):
            """
            Alternating column scanning: start from bottom-left, go upward, then top-down
            x: tensor of shape [B, C, H, W]
            returns: tensor of shape [B, C, H*W]
            """
            B, C, H, W = x.shape
            cols = []
            for i in range(W):
                if i % 2 == 0:  # Even columns - bottom to top
                    cols.append(x[:, :, :, i].flip(dims=[-1]))
                else:  # Odd columns - top to bottom
                    cols.append(x[:, :, :, i])
            return torch.cat([c.view(B, C, -1) for c in cols], dim=-1)

        def diagonal_scan(self, x):
            """
            Diagonal scanning: from bottom-right corner to top-left corner
            x: tensor of shape [B, C, H, W]
            returns: tensor of shape [B, C, H*W]
            """
            B, C, H, W = x.shape
            result = []

            # Start from bottom-right and move toward top-left
            for s in range(H + W - 1, 0, -1):  # Sum of coordinates
                for i in range(H):
                    j = s - i - 1
                    if 0 <= j < W:  # Valid column index
                        result.append(x[:, :, i, j].unsqueeze(-1))

            return torch.cat(result, dim=-1)

        def inverse_z_scan(self, y, H, W):
            """
            Convert back from Z-scan to 2D tensor
            y: tensor of shape [B, H*W, C]
            returns: tensor of shape [B, H, W, C]
            """
            B, L, C = y.shape
            return y.view(B, H, W, C)

        def inverse_s_scan(self, y, H, W):
            """
            Convert back from S-scan to 2D tensor
            y: tensor of shape [B, H*W, C]
            returns: tensor of shape [B, H, W, C]
            """
            B, L, C = y.shape
            result = torch.zeros(B, H, W, C, device=y.device)
            pos = 0

            for i in range(H):
                if i % 2 == 0:  # Even rows (right to left)
                    row = y[:, pos:pos + W, :].flip(dims=[1])
                else:  # Odd rows (left to right)
                    row = y[:, pos:pos + W, :]

                result[:, i, :, :] = row
                pos += W

            return result

        def inverse_alt_column_scan(self, y, H, W):
            """
            Convert back from alternating column scan to 2D tensor
            y: tensor of shape [B, H*W, C]
            returns: tensor of shape [B, H, W, C]
            """
            B, L, C = y.shape
            result = torch.zeros(B, H, W, C, device=y.device)
            pos = 0

            for i in range(W):
                if i % 2 == 0:  # Even columns (bottom to top)
                    col = y[:, pos:pos + H, :].flip(dims=[1])
                else:  # Odd columns (top to bottom)
                    col = y[:, pos:pos + H, :]

                result[:, :, i, :] = col
                pos += H

            return result

        def inverse_diagonal_scan(self, y, H, W):
            """
            Convert back from diagonal scan to 2D tensor
            y: tensor of shape [B, H*W, C]
            returns: tensor of shape [B, H, W, C]
            """
            B, L, C = y.shape
            result = torch.zeros(B, H, W, C, device=y.device)
            pos = 0

            # Start from bottom-right and move toward top-left
            for s in range(H + W - 1, 0, -1):  # Sum of coordinates
                for i in range(H):
                    j = s - i - 1
                    if 0 <= j < W:  # Valid column index
                        result[:, i, j, :] = y[:, pos, :]
                        pos += 1

            return result

        def selective_scan_flatten(self, x, x_proj_weight, x_proj_bias, dt_projs_weight, dt_projs_bias,
                                   A_logs, Ds, out_norm, nrows=1, delta_softplus=True, to_dtype=True, force_fp32=True):
            """
            Apply selective scan operation on input tensor
            Integrated with Mamba's SelectiveScan implementation
            """
            B, L, D = x.shape
            D, N = A_logs.shape
            K, D, R = dt_projs_weight.shape

            # Prepare inputs
            xs = x.transpose(dim0=1, dim1=2).unsqueeze(1).contiguous()

            # Project inputs
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            # Reshape for selective scan
            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
            Bs = Bs.contiguous()
            Cs = Cs.contiguous()
            Ds = Ds.to(torch.float)  # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs = xs.to(torch.float)
                dts = dts.to(torch.float)
                Bs = Bs.to(torch.float)
                Cs = Cs.to(torch.float)

            # 调用Mamba的SelectiveScan实现
            ys = MambaModule.SelectiveScan.apply(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows
            )

            # Reshape output
            ys = ys.view(B, K, -1, L)
            y = ys.squeeze(1).transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)

            # Apply layer normalization
            y = out_norm(y)

            return y.to(x.dtype) if to_dtype else y

        def _forward_impl(self, x):
            # 保存输入用于最终的残差连接
            residual = x

            # Input projection - 不再分割为特征和门控
            x = self.in_proj(x)
            b, h, w, d = x.shape

            # 准备特征图
            x = x.permute(0, 3, 1, 2).contiguous()  # (b, d, h, w)
            x_b0, x_b1 = x[:, :self.b0_dim], x[:, self.b0_dim:]

            # 应用GBC模块替代原来的卷积
            x_b0 = self.gbc_b0(x_b0)  # 全分辨率分支
            x_b1 = self.gbc_b1(x_b1)  # 下采样分支，步长为2

            # 应用SiLU激活函数
            x_b0 = self.act(x_b0)
            x_b1 = self.act(x_b1)

            # 获取下采样分支的尺寸
            h_b1, w_b1 = x_b1.shape[2:]

            # 应用自定义扫描模式
            # 分支0：Z型扫描（单一扫描方向）
            x_b0_scan = self.z_scan(x_b0)

            # 分支1：三种不同的扫描模式
            x_b1_s_scan = self.s_scan(x_b1)  # S型扫描
            x_b1_alt_column = self.alt_column_scan(x_b1)  # 交替列扫描
            x_b1_diagonal = self.diagonal_scan(x_b1)  # 对角线扫描

            # 预处理扫描输入
            scan_b0 = x_b0_scan.permute(0, 2, 1).contiguous()  # (B, L, C)

            scan_b1_s = x_b1_s_scan.permute(0, 2, 1).contiguous()
            scan_b1_col = x_b1_alt_column.permute(0, 2, 1).contiguous()
            scan_b1_diag = x_b1_diagonal.permute(0, 2, 1).contiguous()

            # 应用选择性扫描到每个扫描模式
            # 分支0
            y_b0 = self.selective_scan_flatten(
                scan_b0,
                self.x_proj_weight_b0,
                None,
                self.dt_projs_weight_b0,
                self.dt_projs_bias_b0,
                self.A_logs_b0,
                self.Ds_b0,
                self.out_norm_b0,
                nrows=1,
                delta_softplus=True,
                force_fp32=True
            )

            # 分支1
            y_b1_s = self.selective_scan_flatten(
                scan_b1_s,
                self.x_proj_weight_b1[0:1],
                None,
                self.dt_projs_weight_b1[0:1],
                self.dt_projs_bias_b1[0:1],
                self.A_logs_b1[0:self.b1_dim],
                self.Ds_b1[0:self.b1_dim],
                self.out_norm_b1,
                nrows=1,
                delta_softplus=True,
                force_fp32=True
            )

            y_b1_col = self.selective_scan_flatten(
                scan_b1_col,
                self.x_proj_weight_b1[1:2],
                None,
                self.dt_projs_weight_b1[1:2],
                self.dt_projs_bias_b1[1:2],
                self.A_logs_b1[self.b1_dim:2 * self.b1_dim],
                self.Ds_b1[self.b1_dim:2 * self.b1_dim],
                self.out_norm_b1,
                nrows=1,
                delta_softplus=True,
                force_fp32=True
            )

            y_b1_diag = self.selective_scan_flatten(
                scan_b1_diag,
                self.x_proj_weight_b1[2:3],
                None,
                self.dt_projs_weight_b1[2:3],
                self.dt_projs_bias_b1[2:3],
                self.A_logs_b1[2 * self.b1_dim:3 * self.b1_dim],
                self.Ds_b1[2 * self.b1_dim:3 * self.b1_dim],
                self.out_norm_b1,
                nrows=1,
                delta_softplus=True,
                force_fp32=True
            )

            # 转换回2D形式
            y_b0_2d = self.inverse_z_scan(y_b0, h, w)

            y_b1_s_2d = self.inverse_s_scan(y_b1_s, h_b1, w_b1)
            y_b1_col_2d = self.inverse_alt_column_scan(y_b1_col, h_b1, w_b1)
            y_b1_diag_2d = self.inverse_diagonal_scan(y_b1_diag, h_b1, w_b1)

            # 将分支1的结果转换为张量并合并
            y_b1_s_tensor = y_b1_s_2d.permute(0, 3, 1, 2)
            y_b1_col_tensor = y_b1_col_2d.permute(0, 3, 1, 2)
            y_b1_diag_tensor = y_b1_diag_2d.permute(0, 3, 1, 2)

            # 合并分支1的输出
            y_b1_merged = y_b1_s_tensor + y_b1_col_tensor + y_b1_diag_tensor

            # 上采样分支1以匹配分支0的分辨率
            y_b1_upsampled = F.interpolate(y_b1_merged, size=(h, w), mode='bilinear', align_corners=False)
            y_b1_upsampled = y_b1_upsampled.permute(0, 2, 3, 1).contiguous()

            # 将分支0的结果转换为与分支1相同的格式
            y_b0_tensor = y_b0_2d.permute(0, 3, 1, 2).contiguous()
            y_b0_2d = y_b0_tensor.permute(0, 2, 3, 1).contiguous()

            # 组合分支
            y = torch.cat([y_b0_2d, y_b1_upsampled], dim=-1)

            # 输出投影
            out = self.dropout(self.out_proj(y))

            # 添加全局残差连接
            return out + residual

        def forward(self, x):
            if x.dim() == 4 and x.shape[1] <= self.d_model:
                # 将[B,C,H,W]转换为[B,H,W,C]
                x = x.permute(0, 2, 3, 1)

                # 原有的处理逻辑
                if self.use_checkpoint and self.training:
                    out = torch.utils.checkpoint.checkpoint(self._forward_impl, x)
                else:
                    out = self._forward_impl(x)

                # 将结果从[B,H,W,C]转换回[B,C,H,W]
                out = out.permute(0, 3, 1, 2)

                return out
            else:
                # 原始处理方式
                if self.use_checkpoint and self.training:
                    return torch.utils.checkpoint.checkpoint(self._forward_impl, x)
                else:
                    return self._forward_impl(x)


class MCAModule:
    """
    MCA (Moment Channel Attention) 模块的封装类，包含所有MCA相关组件
    """

    class Moment_efficient(nn.Module):
        """高效的矩特征提取模块，提取均值和标准差"""

        def forward(self, x):
            avg_x = torch.mean(x, (2, 3)).unsqueeze(-1).permute(0, 2, 1)
            std_x = torch.std(x, (2, 3), unbiased=False).unsqueeze(-1).permute(0, 2, 1)
            moment_x = torch.cat((avg_x, std_x), dim=1)  # bs,*,c
            return moment_x

    class Moment_Strong(nn.Module):
        """增强的矩特征提取模块，提取均值和偏度"""

        def forward(self, x):
            # mean std
            n = x.shape[2] * x.shape[3]
            avg_x = torch.mean(x, (2, 3), keepdim=True)  # bs,c,1,1
            std_x = torch.std(x, (2, 3), unbiased=False, keepdim=True)  # bs,c,1,1
            # skew
            skew_x1 = 1 / n * (x - avg_x) ** 3
            skew_x2 = std_x ** 3
            skew_x = torch.sum(skew_x1, (2, 3), keepdim=True) / (skew_x2 + 1e-5)  # bs,c,1,1

            avg_x = avg_x.squeeze(-1).permute(0, 2, 1)
            skew_x = skew_x.squeeze(-1).permute(0, 2, 1)

            moment_x = torch.cat((avg_x, skew_x), dim=1)  # bs,*,c
            return moment_x

    class ChannelAttention(nn.Module):
        """通道注意力模块，通过1D卷积生成通道权重"""

        def __init__(self):
            super(MCAModule.ChannelAttention, self).__init__()
            k = 3  # for Moment_Strong, k = 7
            self.conv = nn.Conv1d(2, 1, kernel_size=k, stride=1, padding=(k - 1) // 2)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            y = self.conv(x)
            output = self.sigmoid(y)
            return output

    class MomentAttention_v1(nn.Module):
        """矩注意力模块V1，使用高效矩特征提取"""

        def __init__(self, **kwargs):
            super(MCAModule.MomentAttention_v1, self).__init__()
            self.moment = MCAModule.Moment_efficient()
            self.c = MCAModule.ChannelAttention()

        def forward(self, x):
            y = self.moment(x)  # bs,2,c
            result = self.c(y)  # bs,1,c
            result = result.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
            return x * result.expand_as(x)

    class MomentAttention_v2(nn.Module):
        """矩注意力模块V2，使用增强矩特征提取"""

        def __init__(self, **kwargs):
            super(MCAModule.MomentAttention_v2, self).__init__()
            self.moment = MCAModule.Moment_Strong()
            self.c = MCAModule.ChannelAttention()

        def forward(self, x):
            y = self.moment(x)  # bs,2,c
            result = self.c(y)  # bs,1,c
            result = result.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
            return x * result.expand_as(x)


class HybridNetworkModule:
    """
    混合神经网络模块的封装类，包含所有网络相关组件
    """

    class HybridNetwork(nn.Module):
        """
        混合神经网络架构，包含AWT卷积模块、Mamba模块和MCA模块
        遵循如下结构:
        输入 -> 分割 -> [AWTConv分支, Mamba分支] -> GELU激活 -> 通道合并 -> MCA -> 输出
        """

        def __init__(
                self,
                in_channels: int,
                hidden_channels: int,
                out_channels: int,
                mamba_d_state: int = 16,
                mamba_d_conv: int = 3,
                mamba_expand_ratio: float = 2.0,
                awt_kernel_size: int = 5,
                awt_wt_levels: int = 1,
                awt_wt_type: str = 'db1',
                dropout: float = 0.0,
                use_checkpoint: bool = False
        ):
            """
            初始化HybridNetwork

            参数:
                in_channels (int): 输入通道数
                hidden_channels (int): 隐藏层通道数
                out_channels (int): 输出通道数
                mamba_d_state (int): Mamba状态维度
                mamba_d_conv (int): Mamba卷积核大小
                mamba_expand_ratio (float): Mamba扩展比率
                awt_kernel_size (int): AWT卷积核大小
                awt_wt_levels (int): 小波变换层级数
                awt_wt_type (str): 小波类型
                dropout (float): Dropout概率
                use_checkpoint (bool): 是否使用checkpoint以节省内存
            """
            super(HybridNetworkModule.HybridNetwork, self).__init__()

            # 计算分支通道数 (将通道平均分配给两个分支)
            self.branch_channels = hidden_channels // 2

            # 输入投影层：将输入通道映射到隐藏通道
            self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)

            # AWT卷积分支
            self.awt_branch = AWTConvModule.WTConv2d(
                in_channels=self.branch_channels,
                out_channels=self.branch_channels,
                kernel_size=awt_kernel_size,
                stride=1,
                bias=True,
                wt_levels=awt_wt_levels,
                wt_type=awt_wt_type
            )

            # Mamba模块分支
            self.mamba_branch = MambaModule.CustomScanMS2D(
                d_model=self.branch_channels,
                d_state=mamba_d_state,
                ssm_ratio=mamba_expand_ratio,
                d_conv=mamba_d_conv,
                dropout=dropout,
                use_checkpoint=use_checkpoint
            )

            # GELU激活函数
            self.gelu = nn.GELU()

            # MCA注意力模块
            self.mca = MCAModule.MomentAttention_v1()

            # 输出投影层
            self.output_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True)

            # 初始化权重
            self._init_weights()

        def _init_weights(self):
            """初始化网络权重"""
            # 卷积层初始化
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            前向传播

            参数:
                x (torch.Tensor): 输入张量，形状为 [B, C, H, W]

            返回:
                torch.Tensor: 输出张量，形状为 [B, out_channels, H, W]
            """
            # 输入投影
            x = self.input_proj(x)

            # 分割通道维度
            x1, x2 = torch.split(x, [self.branch_channels, self.branch_channels], dim=1)

            # 分支处理
            # AWT分支
            x1 = self.awt_branch(x1)
            x1 = self.gelu(x1)

            # Mamba分支
            x2 = self.mamba_branch(x2)
            x2 = self.gelu(x2)

            # 合并分支
            x = torch.cat([x1, x2], dim=1)

            # 应用MCA注意力
            x = self.mca(x)

            # 输出投影
            x = self.output_proj(x)

            return x

    class HybridNetworkBlock(nn.Module):
        """
        混合网络构建块，可用于构建更深层次的网络
        """

        def __init__(
                self,
                channels: int,
                num_blocks: int = 1,
                mamba_d_state: int = 16,
                mamba_d_conv: int = 3,
                mamba_expand_ratio: float = 2.0,
                awt_kernel_size: int = 5,
                awt_wt_levels: int = 1,
                awt_wt_type: str = 'db1',
                dropout: float = 0.0,
                use_checkpoint: bool = False
        ):
            """
            初始化HybridNetworkBlock

            参数:
                channels (int): 通道数
                num_blocks (int): 块的数量
                其他参数与HybridNetwork相同
            """
            super(HybridNetworkModule.HybridNetworkBlock, self).__init__()

            blocks = []
            for _ in range(num_blocks):
                blocks.append(
                    HybridNetworkModule.HybridNetwork(
                        in_channels=channels,
                        hidden_channels=channels * 2,  # 扩展通道
                        out_channels=channels,
                        mamba_d_state=mamba_d_state,
                        mamba_d_conv=mamba_d_conv,
                        mamba_expand_ratio=mamba_expand_ratio,
                        awt_kernel_size=awt_kernel_size,
                        awt_wt_levels=awt_wt_levels,
                        awt_wt_type=awt_wt_type,
                        dropout=dropout,
                        use_checkpoint=use_checkpoint
                    )
                )

            self.blocks = nn.Sequential(*blocks)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            前向传播

            参数:
                x (torch.Tensor): 输入张量

            返回:
                torch.Tensor: 输出张量
            """
            return self.blocks(x)

    class HybridNetworkModel(nn.Module):
        """
        完整的混合网络模型，可用于图像处理任务
        """

        def __init__(
                self,
                in_channels: int,
                out_channels: int,
                base_channels: int = 64,
                num_blocks: List[int] = [2, 2, 6, 2],
                mamba_d_state: int = 16,
                mamba_d_conv: int = 3,
                mamba_expand_ratio: float = 2.0,
                awt_kernel_size: int = 5,
                awt_wt_levels: int = 1,
                awt_wt_type: str = 'db1',
                dropout: float = 0.0,
                use_checkpoint: bool = False
        ):
            """
            初始化HybridNetworkModel

            参数:
                in_channels (int): 输入通道数
                out_channels (int): 输出通道数
                base_channels (int): 基础通道数
                num_blocks (List[int]): 每个阶段的块数量
                其他参数与HybridNetwork相同
            """
            super(HybridNetworkModule.HybridNetworkModel, self).__init__()

            self.input_conv = nn.Conv2d(
                in_channels, base_channels, kernel_size=7, stride=2,
                padding=3, bias=False
            )
            self.norm1 = nn.BatchNorm2d(base_channels)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # 构建网络阶段
            self.stages = nn.ModuleList()
            curr_channels = base_channels

            for i, num_block in enumerate(num_blocks):
                if i > 0:
                    # 下采样层
                    downsample = nn.Sequential(
                        nn.Conv2d(curr_channels, curr_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(curr_channels * 2)
                    )
                    self.stages.append(downsample)
                    curr_channels *= 2

                # 添加块
                self.stages.append(
                    HybridNetworkModule.HybridNetworkBlock(
                        channels=curr_channels,
                        num_blocks=num_block,
                        mamba_d_state=mamba_d_state,
                        mamba_d_conv=mamba_d_conv,
                        mamba_expand_ratio=mamba_expand_ratio,
                        awt_kernel_size=awt_kernel_size,
                        awt_wt_levels=awt_wt_levels,
                        awt_wt_type=awt_wt_type,
                        dropout=dropout,
                        use_checkpoint=use_checkpoint
                    )
                )

            # 输出层
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(curr_channels, out_channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            前向传播

            参数:
                x (torch.Tensor): 输入张量

            返回:
                torch.Tensor: 输出张量
            """
            x = self.input_conv(x)
            x = self.norm1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            for stage in self.stages:
                x = stage(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

    @staticmethod
    def build_hybrid_model(
            in_channels: int = 3,
            out_channels: int = 1000,  # 对于分类任务，通常为类别数
            base_channels: int = 64,
            num_blocks: List[int] = [2, 2, 6, 2],
            **kwargs
    ) -> nn.Module:
        """
        构建混合网络模型

        参数:
            in_channels (int): 输入通道数，默认为3（RGB图像）
            out_channels (int): 输出通道数，默认为1000（类别数）
            base_channels (int): 基础通道数
            num_blocks (List[int]): 每个阶段的块数量
            **kwargs: 其他参数

        返回:
            nn.Module: 混合网络模型
        """
        return HybridNetworkModule.HybridNetworkModel(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            num_blocks=num_blocks,
            **kwargs
        )