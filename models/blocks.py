import torch
import torch.nn as nn


class Conv2dNormActivation(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=None,
            groups=1,
            dilation=1,
            inplace=True,
            bias=None,
            conv_layer=nn.Conv2d,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.ReLU
    ):
        if padding is None:
            padding = (in_channels * (stride - 1) - stride + dilation * (kernel_size - 1) + 1) // 2
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
            )
        ]

        if norm_layer:
            layers.append(norm_layer(out_channels))

        if act_layer:
            layers.append(act_layer(inplace))

        super().__init__(*layers)
        self.out_channels = out_channels


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
    ):
        super().__init__()

        self.conv_block = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding
        )

        if in_channels != out_channels or stride != 1:
            self.res_path = True
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.res_path = False

        self.res = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x_main = self.conv_block(x)
        x_res = self.res_conv(x) if self.res_path else x
        x_out = self.res(x_main + x_res)

        return x_out




