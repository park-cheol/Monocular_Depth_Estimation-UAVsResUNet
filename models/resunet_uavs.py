import torch.nn as nn
import torch

from models.modules import *
from models.attention import *


class ResUnetUAVs(nn.Module):
    def __init__(self, args, channel, filters=[64, 128, 256, 512]):
        super(ResUnetUAVs, self).__init__()
        self.args = args

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )
        self.squeeze_excite0 = Squeeze_Excite_Block(filters[0])

        self.residual_conv_1 = ResidualDepthwiseConv(filters[0], filters[1], 2, 1)
        self.squeeze_excite1 = Squeeze_Excite_Block(filters[1])
        self.residual_conv_2 = ResidualDepthwiseConv(filters[1], filters[2], 2, 1)
        self.squeeze_excite2 = Squeeze_Excite_Block(filters[2])

        self.bridge = ResidualDepthwiseConv(filters[2], filters[3], 2, 1)
        self.attn = Attention(dim=filters[3], heads=8)
        self.aspp = ASPP(filters[3], out_channels=filters[3])

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x1 = self.squeeze_excite0(x1)

        x2 = self.residual_conv_1(x1)
        x2 = self.squeeze_excite1(x2)

        x3 = self.residual_conv_2(x2)
        x3 = self.squeeze_excite2(x3)
        # Bridge
        x4 = self.bridge(x3)
        x4 = self.attn(x4)
        x4 = self.aspp(x4)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10) * self.args.max_depth

        return output


if __name__ == "__main__":
    t = torch.randn(2, 3, 256, 256).cuda()
    g = ResUnetUAVs(args=None, channel=3).cuda()
    parm = sum(p.numel() for p in g.parameters() if p.requires_grad)
    print(parm)
    print(g(t).size())