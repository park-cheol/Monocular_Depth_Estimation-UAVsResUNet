import torch.nn as nn
import torch.nn.functional as F
import torch


class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True,
                 dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, dilation=dilation,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = self.pointwise(out)

        return out


class ResidualDepthwiseConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualDepthwiseConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.LeakyReLU(inplace=True),
            DepthwiseSeparableConv(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(inplace=True),
            DepthwiseSeparableConv(output_dim, output_dim, kernel_size=3, padding=1)
        )
        self.conv_skip = nn.Sequential(
            DepthwiseSeparableConv(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim))

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=8):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   DepthwiseSeparableConv(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        out = self.layer(x)
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):

    def __init__(self, in_channels: int, atrous_rates=[6, 12, 18], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(nn.Sequential(DepthwiseSeparableConv(in_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.LeakyReLU(inplace=True)))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Dropout(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [DepthwiseSeparableConv(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                   nn.BatchNorm2d(out_channels),
                   nn.LeakyReLU(True)]
        super().__init__(*modules)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2
