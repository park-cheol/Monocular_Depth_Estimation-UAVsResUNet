import torch
import torch.nn as nn

from models.modules import ResidualConv, Upsample


class Unet(nn.Module):
    def __init__(self, args, channel, filters=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.args = args

        self.input_layer = nn.Sequential(nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
                                         nn.BatchNorm2d(filters[0]),
                                         nn.ReLU(inplace=True))

        self.conv_1 = nn.Sequential(nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(filters[1]),
                                    nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(filters[2]),
                                    nn.ReLU(inplace=True))

        self.bridge = nn.Sequential(nn.Conv2d(filters[2], filters[3], kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(filters[3]),
                                    nn.ReLU(inplace=True))

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_conv_1 = nn.Sequential(nn.Conv2d(filters[3] + filters[2], filters[2], kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(filters[2]),
                                       nn.ReLU(inplace=True))

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_conv_2 = nn.Sequential(nn.Conv2d(filters[2] + filters[1], filters[1], kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(filters[1]),
                                       nn.ReLU(inplace=True))

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_conv_3 = nn.Sequential(nn.Conv2d(filters[0] + filters[1], filters[0], kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(filters[0]),
                                       nn.ReLU(inplace=True))

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x)
        x2 = self.conv_1(x1)
        x3 = self.conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up_conv_1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up_conv_2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        x10 = self.up_conv_3(x9)
        output = self.output_layer(x10) * self.args.max_depth

        return output


if __name__ == "__main__":
    t = torch.randn(2, 3, 416, 544).cuda()
    g = Unet(args=None, channel=3).cuda()
    parm = sum(p.numel() for p in g.parameters() if p.requires_grad)
    print(parm)
    print(g(t).size())
