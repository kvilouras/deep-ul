import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        return super().forward(x).permute(0, 3, 1, 2).contiguous()


class MaskConv2d(nn.Conv2d):
    """
    Masked 2D convolution block as defined in PixelCNN.
    Note that Type `B' mask is practically the same as
    a Type `A' mask with the addition of the current pixel
    per color channel.
    (e.g. in Type `A' masks, the blue (B) channel is conditioned
    on both green (G) and red (R), whereas in Type `B' it is
    conditioned on R, G and also B, i.e. itself).
    """
    def __init__(self, mask_type, *args, conditional_size=None, **kwargs):
        assert mask_type in ['A', 'B']
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

        if conditional_size is not None:
            # this is useful for conditioning inputs to external info, e.g. ground-truth label.
            self.cond_op = nn.Linear(conditional_size, self.out_channels)

    def forward(self, x, cond=None):
        self.weight.data *= self.mask
        out = super().forward(x)

        if cond is not None:
            cond = self.cond_op(cond)
            out = out + cond.view(cond.shape[0], self.out_channels, 1, 1)

        return out

    def create_mask(self, mask_type):
        k = self.kernel_size[0]
        self.mask[:, :, :k // 2] = 1.
        self.mask[:, :, k // 2, :k // 2] = 1.
        if mask_type == 'B':
            self.mask[:, :, k // 2, k // 2] = 1.


class GatedConv2d(nn.Module):
    """
    Implements Gated masked 2d convolutions (to avoid the blind spot
    introduced in vanilla masked 2d convolutions).
    Here, we use two independent stacks, i.e. a horizontal (conditions
    only on current row) and a vertical (conditions on all top pixels) stack.
    """
    def __init__(self, mask_type, in_channels, out_channels, kernel_size=3, padding=1, conditional_size=None):
        assert mask_type in ['A', 'B']
        super().__init__()

        # define vertical/horizontal stacks + their interaction
        self.vertical = nn.Conv2d(
            in_channels, 2 * out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.horizontal = nn.Conv2d(
            in_channels, 2 * out_channels, kernel_size=(1, kernel_size), padding=(0, padding), bias=False
        )
        self.ver_to_hor = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1, padding=0, bias=False)
        self.hor_to_hor = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False)

        # define masks
        self.register_buffer('vmask', torch.zeros_like(self.vertical.weight))
        self.register_buffer('hmask', torch.zeros_like(self.horizontal.weight))

        self.vmask[:, :, :kernel_size // 2 + 1, :] = 1.
        self.hmask[:, :, :, :kernel_size // 2] = 1.
        if mask_type == 'B':
            self.hmask[:, :, :, kernel_size // 2] = 1.

        if conditional_size:
            self.cond_op_v = nn.Linear(conditional_size, 2 * out_channels, bias=False)
            self.cond_op_h = nn.Linear(conditional_size, 2 * out_channels, bias=False)

    def down_shift(self, x):
        x = x[:, :, :-1, :]
        pad = nn.ZeroPad2d((0, 0, 1, 0))
        return pad(x)

    def forward(self, x, cond=None):
        vx, hx_in = x.chunk(2, dim=1)  # split input in two parts
        self.vertical.weight.data *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx = self.vertical(vx)
        hx = self.horizontal(hx_in)
        # allow horizontal stack to see information from the vertical stack
        # (down-shifting the output of the vertical stack prevents the
        # horizontal stack from seeing future pixels, i.e. pixels on the
        # right of the current row)
        hx = hx + self.ver_to_hor(self.down_shift(vx))

        # conditioning (optional)
        if cond is not None:
            vx += self.cond_op_v(cond).view(cond.shape[0], -1, 1, 1)
            hx += self.cond_op_h(cond).view(cond.shape[0], -1, 1, 1)

        vx_1, vx_2 = vx.chunk(2, dim=1)
        v_out = torch.tanh(vx_1) * torch.sigmoid(vx_2)

        hx_1, hx_2 = hx.chunk(2, dim=1)
        h_out = torch.tanh(hx_1) * torch.sigmoid(hx_2)
        h_out = self.hor_to_hor(h_out)
        h_out += hx_in

        return torch.cat((v_out, h_out), dim=1)


class StackLayerNorm(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.layer_norm_h = LayerNorm(n_filters)
        self.layer_norm_v = LayerNorm(n_filters)

    def forward(self, x):
        vx, hx = x.chunk(2, dim=1)
        vx = self.layer_norm_v(vx)
        hx = self.layer_norm_h(hx)

        return torch.cat((vx, hx), dim=1)


class TempPixelCNN(nn.Module):
    """
    Useful only for the blind spot experiment (shown below)
    """
    def __init__(self, n_layers, conv_type=MaskConv2d):
        super().__init__()
        self.conv_type = conv_type
        layers = [
            conv_type("A", in_channels=1, out_channels=1, kernel_size=3, padding=1)
        ]
        for _ in range(n_layers - 1):
            layers.append(conv_type("B", in_channels=1, out_channels=1, kernel_size=3, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.conv_type == GatedConv2d:
            return self.model(torch.cat((x, x), dim=1)).chunk(2, dim=1)[1]
        else:
            return self.model(x)


if __name__ == "__main__":
    # blind spot experiment
    import matplotlib.pyplot as plt
    import numpy as np

    pixel_pos = (5, 5)
    x = torch.randn(1, 1, 10, 10, requires_grad=True)
    for conv_type in [MaskConv2d, GatedConv2d]:
        fig, ax = plt.subplots(3, 1)
        for i, n_layers in enumerate([2, 3, 5]):
            m = TempPixelCNN(n_layers, conv_type=conv_type)
            out = m(x)
            out[0, 0, pixel_pos[0], pixel_pos[1]].backward()
            grad = x.grad.detach().numpy()[0, 0]
            grad = np.abs(grad)
            grad = (grad > 1e-8).astype("float32")
            grad[pixel_pos[0], pixel_pos[1]] = 0.5

            ax[i].imshow(grad)
            ax[i].set_title(f"{conv_type.__name__} - Receptive field from pixel {pixel_pos} with {n_layers} layers")
            x.grad.zero_()
        plt.tight_layout()
        plt.show()
