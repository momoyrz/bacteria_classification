import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_, DropPath
from timm.models.vision_transformer import Mlp
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def window_partition3d(x, window_size):
    """Partition input tensor into non-overlapping 3D windows"""
    B, C, D, H, W = x.shape
    x = x.view(B, C, D // window_size, window_size,
               H // window_size, window_size,
               W // window_size, window_size)
    windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(-1, window_size**3, C)
    return windows

def window_reverse3d(windows, window_size, D, H, W):
    """Reverse 3D window partition"""
    B = int(windows.shape[0] / (D * H * W / (window_size ** 3)))
    x = windows.reshape(B, D // window_size, H // window_size,
                        W // window_size, window_size, window_size,
                        window_size, -1)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, -1, D, H, W)
    return x


def prepare_multimodal_input(mri_scan, pet_scan):
    """
    合并双模态输入

    Args:
    - mri_scan: 3D sMRI tensor [batch_size, 1, depth, height, width]
    - pet_scan: 3D PET tensor [batch_size, 1, depth, height, width]

    Returns:
    - combined_scan: 双模态tensor [batch_size, 2, depth, height, width]
    """
    return torch.cat([mri_scan, pet_scan], dim=1)

 class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=2, in_dim=64, dim=96):
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(in_dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x

class Downsample3D(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        dim_out = dim if keep_dim else 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv3d(dim, dim_out, kernel_size=3, stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        return self.reduction(x)

class ConvBlock3D(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm3d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(dim, eps=1e-5)

        self.layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        if self.layer_scale:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)

        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(nn.Module):
    """MambaVision mixer layer"""

    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=3,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)

        # Initialize dt
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Initialize A and D
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Convolutions
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            padding=d_conv // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            padding=d_conv // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """Forward pass"""
        _, seqlen, _ = hidden_states.shape

        # Project and split into x, z components
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        # Apply selective scan
        A = -torch.exp(self.A_log.float())
        x = F.silu(self.conv1d_x(x))
        z = F.silu(self.conv1d_z(z))

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None
        )

        # Combine outputs and project
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        return out


class Attention(nn.Module):
    """Multi-head Attention block"""

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Use torch's efficient attention implementation
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """MambaVision block combining attention/mamba with MLP"""

    def __init__(
            self,
            dim,
            num_heads,
            counter,
            transformer_blocks,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            layer_scale=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # Choose mixer type based on block position
        if counter in transformer_blocks:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            self.mixer = MambaVisionMixer(
                d_model=dim,
                d_state=8,
                d_conv=3,
                expand=1
            )

        # MLP part
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

        # Layer scaling if needed
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """MambaVision layer grouping multiple blocks"""

    def __init__(
            self,
            dim,
            depth,
            num_heads,
            window_size,
            conv=False,
            downsample=True,
            mlp_ratio=4.,
            qkv_bias=True,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            layer_scale=None,
            layer_scale_conv=None,
            transformer_blocks=[],
    ):
        super().__init__()
        self.conv = conv
        self.transformer_block = not conv

        # Create blocks based on type
        if conv:
            self.blocks = nn.ModuleList([
                ConvBlock(
                    dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale_conv
                ) for i in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=dim,
                    counter=i,
                    transformer_blocks=transformer_blocks,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale
                ) for i in range(depth)
            ])

        # Downsampling
        self.downsample = None if not downsample else Downsample(dim=dim)
        self.window_size = window_size

    def forward(self, x):
        _, _, D, H, W = x.shape  # 修改为3D维度

        # 处理3D窗口
        if self.transformer_block:
            pad_d = (self.window_size - D % self.window_size) % self.window_size
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size

            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
                _, _, Dp, Hp, Wp = x.shape
            else:
                Dp, Hp, Wp = D, H, W

            x = window_partition3d(x, self.window_size)  # 使用3D窗口划分

        # 处理块
        for blk in self.blocks:
            x = blk(x)

        # 恢复3D窗口
        if self.transformer_block:
            x = window_reverse3d(x, self.window_size, Dp, Hp, Wp)
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x = x[:, :, :D, :H, :W].contiguous()

        # 下采样
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


class MambaVision3D(nn.Module):
    """Complete MambaVision 3D model"""
    def __init__(
            self,
            dim=96,  # Base dimension
            in_dim=64,  # Initial embedding dimension
            depths=[2, 2, 4, 2],  # Tiny-2 configuration
            window_size=[8, 8, 8, 8],  # 修改为3D窗口大小
            mlp_ratio=4.,
            num_heads=[3, 6, 12, 24],
            drop_path_rate=0.2,
            in_chans=2,  # 修改为2个模态
            num_classes=2,  # 修改为阿尔兹海默病分类数
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            layer_scale=None,
            layer_scale_conv=None,
    ):
        super().__init__()
        self.num_classes = num_classes

        # 修改为3D补丁嵌入
        self.patch_embed = PatchEmbed3D(
            in_chans=in_chans,
            in_dim=in_dim,
            dim=dim
        )

        # Drop path rates for each block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 保留原有的层创建逻辑
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            # First two levels use ConvBlocks
            conv = True if (i == 0 or i == 1) else False

            # Determine which blocks in each layer should use transformer attention
            transformer_blocks = list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0 else list(
                range(depths[i] // 2, depths[i]))

            level = MambaVisionLayer3D(  # 修改为3D层
                dim=int(dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                conv=conv,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=(i < 3),  # Downsample in first 3 levels
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                transformer_blocks=transformer_blocks,
            )
            self.levels.append(level)

        # Final layers
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.norm = nn.BatchNorm3d(num_features)  # 修改为3D归一化
        self.avgpool = nn.AdaptiveAvgPool3d(1)  # 修改为3D平均池化
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):  # 修改为3D批归一化
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """Extract features from input images"""
        x = self.patch_embed(x)
        outs = []

        for level in self.levels:
            x, xo = level(x)
            outs.append(xo)

        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x, outs

    def forward(self, x):
        """Forward pass"""
        x, _ = self.forward_features(x)
        x = self.head(x)
        return x


def load_pretrained_weights(model, checkpoint_path):
    """
    Helper function to load pretrained weights

    Args:
        model: MambaVision model instance
        checkpoint_path: Path to the checkpoint file

    Returns:
        model: Model with loaded weights
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # Process state dict if needed
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    # Remove module prefix if present
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Handle encoder prefix
    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    model.load_state_dict(state_dict, strict=False)
    return model


def create_mambavision_tiny2(in_chans=2, pretrained=False, pretrained_path=None):
    """
    Create MambaVision model with Tiny-2 configuration

    Args:
        in_chans: Number of input channels (default: 2 for MRI and PET)
        pretrained: Whether to load pretrained weights
        pretrained_path: Path to pretrained weights file

    Returns:
        model: Initialized MambaVision model
    """
    model = MambaVision3D(  # 修改为MambaVision3D
        dim=96,
        in_dim=64,
        depths=[2, 2, 4, 2],  # Tiny-2 configuration
        window_size=[8, 8, 8, 8],  # 3D窗口大小
        mlp_ratio=4,
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.2,
        in_chans=in_chans,  # 2个模态
        num_classes=2,  # 阿尔兹海默病分类
    )

    if pretrained and pretrained_path:
        model = load_pretrained_weights(model, pretrained_path)

    return model