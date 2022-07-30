import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MaskConv2d, LayerNorm


__all__ = [
    'hierarchical_priors'
]


class GatedResBlock(nn.Module):
    """
    Implements the gated residual network block as described
    in PixelSNAIL paper.
    Note that the output's channels are equal to in_channels!

    Args:
        in_channels: Number of input channels
        channels: Number of intermediate channels
        kernel_size: Kernel size
        padding: If set to None, it performs "same" padding,
            i.e. spatial dimensions are preserved the same.
        mask_center: If True, it masks the center pixel, i.e.
            it performs a Type 'A' causal convolution.
        conv_type: Convolution type (either masked or standard)
        activation: Activation function (default: ELU)
        p: Dropout probability
        conditional_size: Size of external inputs used for
            conditioning, e.g. ground-truth labels.
        top_conditional_channel: Number of channels in top-level latent
            codes, only used for the bottom-level prior model.
        auxiliary_channel: Number of channels in auxiliary inputs,
            e.g. attention maps.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_size=3,
            padding=None,
            mask_center=True,
            conv_type=MaskConv2d,
            activation=nn.ELU,
            p=0.1,
            conditional_size=None,
            top_conditional_channel=None,
            auxiliary_channel=0
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        if conv_type == MaskConv2d:
            mask_type = 'A' if mask_center else 'B'
            conv_type = partial(MaskConv2d, mask_type, conditional_size=conditional_size)

        self.conv1 = conv_type(in_channels, channels, kernel_size, padding=padding)
        self.conv2 = conv_type(channels, in_channels * 2, kernel_size, padding=padding)
        self.activation = activation()
        self.conditional_size = conditional_size

        if top_conditional_channel is not None:
            # useful for conditioning on top-level latent codes
            self.cond_op = nn.Conv2d(top_conditional_channel, in_channels * 2, 1, bias=False)

        if auxiliary_channel > 0:
            # for auxiliary inputs, e.g. attention maps
            self.aux_op = nn.Conv2d(auxiliary_channel, channels, 1)

        self.dropout = nn.Dropout(p)
        self.gate = nn.GLU(1)

    def forward(self, x, cond=None, top_cond=None, aux=None):
        out = self.conv1(x, cond=cond) if self.conditional_size else self.conv1(x)
        if aux is not None:
            out += self.aux_op(self.activation(aux))
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out, cond=cond) if self.conditional_size else self.conv2(out)

        if top_cond is not None:
            out += self.cond_op(top_cond)

        return self.gate(out) + x


def gen_positional_encodings(size):
    """
    Generates positional encodings for 2d input images.
    Args:
        size: Image shape (N, C, H, W)

    Returns:
        Tensor of size (N, 2, H, W) with values in [-0.5, 0.5) range
    """

    n, c, h, w = size

    h_enc = torch.arange(h, dtype=torch.float32) / h - 0.5
    h_enc = h_enc.view(1, 1, h, 1).expand(n, 1, -1, w)

    w_enc = torch.arange(w, dtype=torch.float32) / w - 0.5
    w_enc = w_enc.view(1, 1, 1, w).expand(n, 1, h, -1)

    return torch.cat((h_enc, w_enc), dim=1)


class CausalSelfAttention(nn.Module):
    """
    Implements the causal self-attention block used in
    PixelSNAIL. Here, linear layers are replaced by 1x1 convs.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output (aka value) channels
        embed_channels (optional): Number of intermediate (aka key) channels.
            If not given, it defaults to in_channels.
        extra_in_channels: Number of channels contained in extra information.
            Here, we use the original image as additional info to calculate
            the embeddings (keys/values).
        n_heads: Number of attention heads
        mask_center: If True, the center pixel of the attention matrix is masked.
        p: Dropout probability
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            embed_channels=None,
            extra_in_channels=0,
            n_heads=8,
            mask_center=True,
            p=0.1
    ):
        super().__init__()

        if embed_channels is None:
            embed_channels = in_channels
        self.embed_channels = embed_channels
        self.out_channels = out_channels

        self.q_conv = nn.Conv2d(in_channels, embed_channels, 1)
        self.kv_conv = nn.Conv2d(
            in_channels + extra_in_channels, embed_channels + out_channels, 1
        )

        self.n_heads = n_heads
        self.mask_center = mask_center
        self.dropout = nn.Dropout(p)

    def forward(self, x, extra_x=None):
        b, _, h, w = x.shape

        q = self.q_conv(x)
        q = self.to_multihead(q)

        if extra_x is not None:
            x = torch.cat((x, extra_x), dim=1)
        k, v = self.kv_conv(x).split([self.embed_channels, self.out_channels], dim=1)
        k, v = self.to_multihead(k), self.to_multihead(v)

        mask = self.causal_mask(k.shape[-2], self.mask_center).view(1, 1, h * w, h * w)
        mask = mask.to(x.device)

        attn = q @ k.transpose(2, 3) / math.sqrt(k.shape[-1])  # shape: (B, N_heads, H*W, H*W)
        attn.masked_fill_(mask == 0, -1e4)
        attn = F.softmax(attn, dim=-1).masked_fill(mask == 0, 0)
        attn = self.dropout(attn)

        out = attn @ v  # shape: (B, N_heads, H*W, C_out//N_heads)
        out = out.transpose(2, 3).contiguous().view(b, -1, h, w)  # shape: (B, C_out, H, W)

        return out

    @staticmethod
    def causal_mask(size, mask_center):
        return torch.tril(torch.ones(size, size), diagonal=-int(mask_center))

    def to_multihead(self, x):
        """
        Convert input of shape (B, C, H, W) to (B, N_heads, H * W, C // N_heads)
        """

        b, c = x.shape[:2]
        return x.view(b, self.n_heads, c // self.n_heads, -1).transpose(2, 3).contiguous()


class PixelSNAILBlock(nn.Module):
    """
    Basic block of PixelSNAIL architecture
    """

    def __init__(
            self,
            in_channels,
            kernel_size=3,
            padding=None,
            mask_center=True,
            n_res_block=2,
            attention=True,
            attention_key_channels=4,
            attention_value_channels=32,
            input_img_channels=3,
            p=0.1,
            conditional_size=None,
            top_conditional_channel=None,
            norm_layer=LayerNorm,
            act_layer=nn.ELU
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        res_blocks = []
        for _ in range(n_res_block):
            res_blocks.append(
                GatedResBlock(
                    in_channels,
                    in_channels,
                    kernel_size,
                    padding,
                    mask_center,
                    conv_type=MaskConv2d,
                    p=p,
                    conditional_size=conditional_size,
                    top_conditional_channel=top_conditional_channel
                )
            )
            res_blocks.extend([norm_layer(in_channels), act_layer()])
        self.res_blocks = nn.ModuleList(res_blocks)

        self.attention = attention
        if attention:
            self.causal_attention = CausalSelfAttention(
                in_channels=in_channels + 2,
                embed_channels=attention_key_channels,
                out_channels=attention_value_channels,
                extra_in_channels=input_img_channels,
                mask_center=mask_center,
                p=p
            )

            self.out = GatedResBlock(
                in_channels,
                in_channels,
                1,
                0,
                mask_center,
                conv_type=nn.Conv2d,
                p=p,
                auxiliary_channel=attention_value_channels
            )
        else:
            self.out = nn.Conv2d(in_channels + 2, in_channels, 1)

    def forward(self, x, pos_enc, cond=None, top_cond=None):
        out = x
        for b in self.res_blocks:
            if isinstance(b, GatedResBlock):
                out = b(out, cond=cond, top_cond=top_cond)
            else:
                out = b(out)

        if self.attention:
            attn = self.causal_attention(torch.cat((out, pos_enc), dim=1), extra_x=x)
            out = self.out(out, aux=attn)
        else:
            out = torch.cat((out, pos_enc), dim=1)
            out = self.out(out)

        return out


class CondResNet(nn.Module):
    """
    Residual network for the conditioning stack.
    Note that we do not need to use causal convs here
    """

    def __init__(self, in_channels, out_channels, kernel_size, n_blocks, norm_layer=LayerNorm, act_layer=nn.ELU):
        super().__init__()

        padding = kernel_size // 2
        blocks = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        ]
        for _ in range(n_blocks):
            blocks.extend([norm_layer(out_channels), act_layer()])
            blocks.append(
                GatedResBlock(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding,
                    conv_type=nn.Conv2d
                )
            )
        blocks.extend([norm_layer(out_channels), act_layer()])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class PixelSNAIL(nn.Module):
    """
    Implements the PixelSNAIL architecture https://arxiv.org/abs/1712.09763.
    Note that this model is adapted to the version used in VQ-VAE2 paper.

    Args:
        input_shape: Original input's spatial dimensions
        code_size: Codebook size
        dim: Embedding dimension
        kernel_size: Convolutional kernel size
        mask_center: If True, the center pixel is masked (Type 'A' causal conv)
        n_blocks: Total number of PixelSNAIL blocks
        n_res_blocks: Number of residual blocks inside each PixelSNAIL block
        attention: If True, causal attention is used (note that the VQ-VAE2
            paper uses attention only for the top-level prior model)
        attention_key_channels: Number of channels for calculating key embeddings
        attention_value_channels: Number of channels for calculating value embeddings
        p: Dropout probability
        n_cond_res_blocks: Number of residual blocks for the conditioning stack (used
            in bottom-level prior model)
        cond_kernel: Conditioning stack's kernel size
        n_out_res_blocks: Number of output stack residual blocks
        device: Device (either cpu or cuda:0)
        conditional_size: Size of external information (e.g. ground-truth labels) used
            for conditioning.
        top_conditional: If True, it initializes the conditioning stack. This argument
            should only be used for the bottom-level prior model.
        act_layer: Activation layer type
        norm_layer: Normalization layer type
    """

    def __init__(
            self,
            input_shape,
            code_size=128,
            dim=256,
            kernel_size=5,
            mask_center=True,
            n_blocks=2,
            n_res_blocks=5,
            attention=True,
            attention_key_channels=16,
            attention_value_channels=128,
            p=0.1,
            n_cond_res_blocks=0,
            cond_kernel=3,
            n_out_res_blocks=0,
            device=None,
            conditional_size=None,
            top_conditional=False,
            act_layer=nn.ELU,
            norm_layer=LayerNorm
    ):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input_shape = input_shape
        self.code_size = code_size

        self.embedding = nn.Embedding(code_size, dim)
        if top_conditional:
            self.top_embedding = nn.Embedding(code_size, dim)
        self.in_conv = MaskConv2d('A', in_channels=dim, out_channels=dim, kernel_size=7, padding=3,
                                  conditional_size=conditional_size)
        blocks = []
        for i in range(n_blocks):
            blocks.extend([norm_layer(dim), act_layer()])
            blocks.append(
                PixelSNAILBlock(
                    dim,
                    kernel_size,
                    mask_center=mask_center,
                    n_res_block=n_res_blocks,
                    attention=attention,
                    attention_key_channels=attention_key_channels,
                    attention_value_channels=attention_value_channels,
                    input_img_channels=dim,
                    p=p,
                    conditional_size=conditional_size,
                    top_conditional_channel=dim if top_conditional else None,
                    norm_layer=norm_layer,
                    act_layer=act_layer
                )
            )
        blocks.extend([norm_layer(dim), act_layer()])
        self.blocks = nn.ModuleList(blocks)

        # conditioning stack residual blocks
        if n_cond_res_blocks > 0:
            self.cond_resnet = CondResNet(dim, dim, cond_kernel, n_cond_res_blocks, norm_layer, act_layer)

        # output stack residual blocks
        out = []
        for _ in range(n_out_res_blocks):
            out.append(
                GatedResBlock(dim, dim, 1, conv_type=nn.Conv2d, p=p)
            )
            out.extend([norm_layer(dim), act_layer()])
        out.append(
            nn.Conv2d(dim, code_size, 1)
        )
        self.out = nn.Sequential(*out)

    def forward(self, x, cond=None, top_condition=None):
        """
        Args:
            x: Original input of size (N, C, H, W)
            cond: External information (e.g. ground-truth labels),
                useful in the case of a conditional prior model.
            top_condition: Conditioning stack that contains latent
                codes from the top level (used in the bottom-level
                prior model).

        Returns:
            Output tensor (logits) of size (N, code_dim, H, W)
        """

        x = self.embedding(x).permute(0, 3, 1, 2).contiguous()
        b, _, h, w = x.shape
        out = self.in_conv(x, cond=cond)

        pos_enc = gen_positional_encodings(x.shape)

        if top_condition is not None:
            top_condition = self.top_embedding(top_condition).permute(0, 3, 1, 2).contiguous()
            top_condition = self.cond_resnet(top_condition)
            top_condition = F.interpolate(top_condition, scale_factor=2)[:, :, :h, :]

        for block in self.blocks:
            if isinstance(block, PixelSNAILBlock):
                out = block(out, pos_enc, cond=cond, top_cond=top_condition)
            else:
                out = block(out)

        out = self.out(out)

        return out

    def loss(self, x, cond=None, top_condition=None):
        """
        Calculate NLL loss + bits/dim (bpd) metric.
        """

        out = self(x, cond=cond, top_condition=top_condition)
        nll = F.cross_entropy(out, x.long())
        bpd = nll / math.log(2)

        return dict(nll_loss=nll, bpd=bpd)

    def sample(self, n, cond=None, top_cond=None):
        """
        Autoregressive generation
        """

        if cond is not None and cond.device != self.device:
            cond = cond.to(self.device)
        if top_cond is not None and top_cond.device != self.device:
            top_cond = top_cond.to(self.device)
        samples = torch.zeros(n, *self.input_shape, dtype=torch.long, device=self.device)
        with torch.no_grad():
            for h in range(self.input_shape[0]):
                for w in range(self.input_shape[1]):
                    logits = self(samples, cond=cond, top_condition=top_cond)[:, :, h, w]
                    probs = F.softmax(logits, dim=1)
                    samples[:, h, w] = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return samples


def hierarchical_priors(**kwargs):
    """
    Instantiate both a top-level and a bottom-level autoregressive
    prior model (PixelSNAIL).
    """

    n_cond_res_blocks = kwargs.pop('n_cond_res_blocks')
    top_input_shape = kwargs.pop('top_input_shape')
    bottom_input_shape = kwargs.pop('bottom_input_shape')
    top_prior = PixelSNAIL(input_shape=top_input_shape, attention=True, **kwargs)
    bottom_prior = PixelSNAIL(
        input_shape=bottom_input_shape, n_cond_res_blocks=n_cond_res_blocks, top_conditional=True, attention=False,
        **kwargs
    )

    return top_prior, bottom_prior
