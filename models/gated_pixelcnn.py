import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MaskConv2d, GatedConv2d, StackLayerNorm


class GatedPixelCNN(nn.Module):
    """
    Gated PixelCNN network as introduced in https://arxiv.org/pdf/1606.05328v2.pdf

    Args:
        input_shape: Input's spatial dimensions (H, W)
        code_size: Codebook size
        dim: Embedding's dimension
        n_layers: Number of layers
    """

    def __init__(self, input_shape, code_size=128, dim=256, n_layers=7, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.code_size = code_size

        self.embedding = nn.Embedding(code_size, dim)
        self.in_conv = MaskConv2d('A', in_channels=dim, out_channels=dim, kernel_size=7, padding=3)
        layers = []
        for _ in range(n_layers - 2):
            layers.extend(
                [
                    StackLayerNorm(dim),
                    nn.ReLU(),
                    GatedConv2d('B', in_channels=dim, out_channels=dim, kernel_size=7, padding=3)
                ]
            )
        layers.extend([StackLayerNorm(dim), nn.ReLU()])
        self.out_conv = MaskConv2d('B', in_channels=dim, out_channels=code_size, kernel_size=7, padding=3)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.embedding(x).permute(0, 3, 1, 2).contiguous()
        out = self.in_conv(out)
        out = self.model(torch.cat((out, out), dim=1)).chunk(2, dim=1)[1]
        out = self.out_conv(out)

        return out

    def loss(self, x):
        """
        Calculate NLL loss + bits/dim (bpd) metric.

        Bpd describes how many bits are needed to encode an example
        in our modeled distribution. The less bits we need, the more
        likely the example is in our distribution. E.g., for a value
        of bpd=8, we need 8 bits to encode each pixel (i.e. there are
        2 ** 8 = 256 possible values).
        """

        out = self(x)
        nll = F.cross_entropy(out, x.long())
        bpd = nll / math.log(2)

        return dict(nll_loss=nll, bpd=bpd)

    def sample(self, n):
        """
        Autoregressive generation of n samples in total
        """

        samples = torch.zeros(n, *self.input_shape, dtype=torch.long, device=self.device)
        with torch.no_grad():
            for h in range(self.input_shape[0]):
                for w in range(self.input_shape[1]):
                    logits = self(samples)[:, :, h, w]
                    probs = F.softmax(logits, dim=1)
                    samples[:, h, w] = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return samples




