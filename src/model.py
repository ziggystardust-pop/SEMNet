from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x
class BlockwiseSE(nn.Module):
    def __init__(self, channel, B=4, reduction=16):
        super(BlockwiseSE, self).__init__()
        assert channel % B == 0
        self.B = B
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        channel = channel//B
        self.fc = nn.Sequential(
                nn.Linear(channel, max(2, channel//reduction)),
                nn.ReLU(inplace=True),
                nn.Linear(max(2, channel//reduction), channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b*self.B, c//self.B)
        y = self.fc(y).view(b, c, 1, 1)
        return x*y
class BlockwiseSEConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=16):
        super(BlockwiseSEConv, self).__init__()
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn_res = nn.BatchNorm2d(out_channels)
        self.conv = conv_block(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se_block = BlockwiseSE(out_channels, B=4, reduction=reduction_ratio)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.se_block(out)
        residual = self.bn_res(self.residual_conv(residual))
        out += residual
        out = self.relu(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class Blockwise_avg__max(nn.Module):
    def __init__(self, channel, B=4, reduction=16):
        super(Blockwise_avg__max, self).__init__()
        assert channel % B == 0
        self.B = B
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        channel = channel//B
        self.fc = nn.Sequential(
                nn.Linear(channel, max(2, channel//reduction)),
                nn.ReLU(inplace=True),
                nn.Linear(max(2, channel//reduction), channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b*self.B, c//self.B)
        y2 = self.max_pool(x).view(b*self.B, c//self.B)
        y = self.fc(y).view(b, c, 1, 1)
        y2 = self.fc(y2).view(b, c, 1, 1)


        return x*(y+y2)

class AFFM(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels,r_2 = 16):

        super(AFFM, self).__init__()

        self.sa = SpatialAttention()
        self.se = Blockwise_avg__max(in_channels2,4,16)
        self.norm1 = LayerNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.fuse = nn.Sequential(IDSC(out_channels*3, out_channels),
                              nn.BatchNorm2d(out_channels),
                              nn.GELU())
        self.g = nn.GELU()
    def forward(self, out1, out2):
        out1_sa = self.sa(out1)
        x_sig = self.sigmoid(out1_sa)
        out1 = out1 * x_sig
        out1 = self.norm1(out1)
        out2_sa = self.sa(out2)
        y_sig = self.sigmoid(out2_sa)
        out2 = out2 * y_sig
        out2 = self.se(out2)
        out2 = self.norm1(out2)
        out1_g = self.sigmoid(self.g(out1))
        out2_g = self.sigmoid(self.g(out2))
        out3 = out1_g*out2_g
        out3 = self.norm1(out3)
        out = torch.cat([out1, out2,out3], dim=1)
        out = self.fuse(out)

        return out

class SEMNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1):
        super(SEMNet, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ResSEBlock1 = BlockwiseSEConv(in_channels, filters[0])
        self.ResSEBlock2 = BlockwiseSEConv(filters[0], filters[1])
        self.ResSEBlock3 = BlockwiseSEConv(filters[1], filters[2])
        self.ResSEBlock4 = BlockwiseSEConv(filters[2], filters[3])
        self.ResSEBlock5 = BlockwiseSEConv(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.aux_conv5 = nn.Conv2d(filters[3], num_classes, kernel_size=1, stride=1, padding=0)
        self.aux_conv4 = nn.Conv2d(filters[2], num_classes, kernel_size=1, stride=1, padding=0)
        self.aux_conv3 = nn.Conv2d(filters[1], num_classes, kernel_size=1, stride=1, padding=0)

        self.fusion2 = AFFM(filters[1],filters[1],filters[1])
        self.fusion3 = AFFM(filters[2],filters[2],filters[2])
        self.fusion4 = AFFM(filters[3],filters[3],filters[3])
        self.fusion5 = AFFM(filters[4],filters[4],filters[4])

        self.tfstage = Transformer()

    def forward(self, x):
        _, _, h, w = x.size()
        tformerstages = self.tfstage(x)
        e1 = self.ResSEBlock1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.ResSEBlock2(e2)
        e2_t = self.fusion2(e2, tformerstages[0])
        e3 = self.Maxpool2(e2)
        e3 = self.ResSEBlock3(e3)
        e3_t = self.fusion3(e3, tformerstages[1])
        e4 = self.Maxpool3(e3)
        e4 = self.ResSEBlock4(e4)
        e4_t = self.fusion4(e4, tformerstages[2])
        e5 = self.Maxpool4(e4)
        e5 = self.ResSEBlock5(e5)
        e5_t = self.fusion5(e5, tformerstages[3])
        d5 = self.Up5(e5_t)
        x4 = self.Att5(d5, e4_t)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        x3 = self.Att4(d4, e3_t)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        x2 = self.Att3(d3, e2_t)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        x1 = self.Att2(d2, e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        logits = self.Conv(d2)
        aux_out5 = self.aux_conv5(d5)
        aux_out5 = F.interpolate(aux_out5, size=(h, w), mode='bilinear',
                                 align_corners=True)
        aux_out4 = self.aux_conv4(d4)
        aux_out4 = F.interpolate(aux_out4, size=(h, w), mode='bilinear',
                                 align_corners=True)
        aux_out3 = self.aux_conv3(d3)
        aux_out3 = F.interpolate(aux_out3, size=(h, w), mode='bilinear',
                                 align_corners=True)
        return [logits, aux_out5, aux_out4, aux_out3]

class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, dila=2, padding=1):
        super(DSC, self).__init__()
        # self.depthwise = nn.Conv2d(c_in, c_in, k_size, stride=1, padding=padding, dilation=dila, groups=c_in)

        # # 逐点卷积 (Pointwise Convolution)
        # self.pointwise = nn.Conv2d(in_channels, out_channels, 1,1)
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out


class PatchEmbed(nn.Module):
    def __init__(self, dim, p_size):
        super().__init__()
        self.embed = DSC(3, dim, p_size, p_size, 0)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.norm(self.embed(x))
        return x


class PatchMerge(nn.Module):
    def __init__(self, inc, outc, kernel_size=2):
        super().__init__()
        self.merge = DSC(inc, outc, k_size=kernel_size, stride=kernel_size, padding=0)
        self.norm = nn.BatchNorm2d(outc)

    def forward(self, x):
        return self.norm(self.merge(x))


from torch.cuda.amp import autocast
from inspect import signature
from functools import partial
from typing import Dict, Tuple


def get_same_padding(kernel_size: int or Tuple[int, ...]) -> int or Tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def val2list(x: list or tuple or any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def build_kwargs_from_config(config: dict, target_func: callable) -> Dict[str, any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


# register activation function here
REGISTERED_ACT_DICT: Dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}


def build_act(name: str, **kwargs) -> nn.Module or None:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


# register normalization function here
REGISTERED_NORM_DICT: Dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> nn.Module or None:
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, dilation=1, groups=1,
                 use_bias=False, dropout=0, norm="bn2d", act_func="relu", ):
        super(ConvLayer, self).__init__()
        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                              stride=(stride, stride), padding=padding,
                              dilation=(dilation, dilation), groups=groups, bias=use_bias, )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class MLA(nn.Module):

    def __init__(
            self,
            in_channels: int, out_channels: int, heads: int or None = None, heads_ratio: float = 1.0, dim=8,
            use_bias=False,
            norm=(None, "bn2d"), act_func=(None, None), kernel_func="relu", scales: Tuple[int, ...] = (5,),
            eps=1.0e-15, ):
        super(MLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(in_channels, 3 * total_dim, 1, use_bias=use_bias[0], norm=norm[0], act_func=act_func[0], )
        self.aggreg = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim,
                          bias=use_bias[0], ),
                nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
            )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)
        self.proj = ConvLayer(total_dim * (1 + len(scales)), out_channels, 1, use_bias=use_bias[1], norm=norm[1],
                              act_func=act_func[1], )

    @autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W,), )
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (qkv[..., 0: self.dim], qkv[..., self.dim: 2 * self.dim], qkv[..., 2 * self.dim:],)

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)
        out = self.relu_linear_att(multi_scale_qkv)
        out = self.proj(out)
        return out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        # 直接使用输入形状，不需要显式计算 H 和 W
        x = self.dwconv(x)
        return x

import numbers

class DWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
    def forward(self, x):
        return x + self.conv(x)

class RDWCFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True, downsample=False, kernel_size=5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
               
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
        self.conv = DWC(hidden_features, 3)
        
    def forward(self, x):       
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, inc, num_head=8,dropout=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(inc)
        self.norm2 = nn.BatchNorm2d(inc)
        self.attn = MLA(in_channels=inc, out_channels=inc, heads=num_head, scales=(3,5,7))
        self.mlp = RDWCFFN(inc, drop=dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        # layer1
        self.embed = PatchEmbed(128, 4)
        self.block1 = Block(128, num_head=8)
        # layer2
        self.merge2 = PatchMerge(128, 256)
        self.block2 = Block(256, num_head=8)
        # layer3
        self.merge3 = PatchMerge(256, 512)
        self.block3 = Block(512, num_head=8)
        # layer4
        self.merge4 = PatchMerge(512, 1024)
        self.block4 = Block(1024, num_head=8)

    def forward(self, x):
        # layer1
        x1 = self.embed(x)
        x1 = self.block1(x1)
        # layer2
        x2 = self.merge2(x1)
        x2 = self.block2(x2)
        # layer3
        x3 = self.merge3(x2)
        x3 = self.block3(x3)
        # layer4
        x4 = self.merge4(x3)
        out = self.block4(x4)
        out_up = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        x3_up = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x2_up = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x1_up = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        return x1_up, x2_up, x3_up, out_up
