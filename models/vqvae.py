import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv2dNormActivation, ResBlock


__all__ = [
    'VQVAE',
    'VQVAE2'
]


class Quantize(nn.Module):
    """
    Implements the quantization step in VQ-VAE.
    Here, the embeddings are learnable! (i.e., no EMA updates are involved)

    Args:
        code_size: Size of the codebook, i.e. the total number of latents
        code_dim: Embedding's dimension
    """

    def __init__(self, code_dim, code_size):
        super().__init__()
        self.embeddings = nn.Embedding(code_size, code_dim)
        # initialize embeddings to Uniform(-1/K, 1/K), K: code size
        self.embeddings.weight.data.uniform_(-1. / code_size, 1. / code_size)

        self.code_dim = code_dim
        self.code_size = code_size

    def forward(self, x):
        """
        Args:
            x: Feature maps (encoder's output) of size (B, C, H, W)

        Returns:
            Latent codes of size (B, C, H, W)
            Straight-through estimator to propagate gradients from the decoder to the encoder
                (vq is non-differentiable). Note that this serves as the decoder's input
            Encoding indices of size (B, H, W)
        """

        b, c, h, w = x.shape
        weight = self.embeddings.weight

        flat_inputs = x.permute(0, 2, 3, 1).contiguous().view(-1, self.code_dim)
        # measure distances from all codes
        dists = torch.cdist(flat_inputs, weight, p=2)
        idxs = torch.max(-dists, dim=1)[1].view(b, h, w)
        quantized = self.embeddings(idxs).permute(0, 3, 1, 2).contiguous()

        return quantized, (quantized - x).detach() + x, idxs


class VQVAE(nn.Module):
    """
    Implements the Vector Quantized Variational AutoEncoder (VQ-VAE) model

    Args:
        out_channels: Number of output feature maps (for the encoder)
        num_blocks: Number of residual blocks (both in encoder and decoder)
        code_dim: Embedding's dimension
        code_size: Codebook's length
        beta: Beta coefficient in VQ-VAE loss (used for disentanglement)
    """

    def __init__(self, out_channels=256, num_blocks=2, code_dim=256, code_size=128, beta=1.):
        super().__init__()
        # make sure the number of channels in the encoder's last layer is equal to code_dim!!!
        assert out_channels == code_dim
        self.beta = beta

        # Encoder's architecture
        layers = [
            Conv2dNormActivation(3, out_channels, kernel_size=4, stride=2, padding=1),
            Conv2dNormActivation(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ]
        for _ in range(num_blocks):
            layers.append(
                ResBlock(out_channels, out_channels)
            )
        self.encoder = nn.Sequential(*layers)

        # Codebook
        self.codebook = Quantize(code_dim, code_size)

        # Decoder's architecture
        layers = []
        for _ in range(num_blocks):
            layers.append(
                ResBlock(out_channels, out_channels)
            )
        layers.extend(
            [
                Conv2dNormActivation(
                    out_channels, out_channels, kernel_size=4, stride=2, padding=1, conv_layer=nn.ConvTranspose2d
                ),
                Conv2dNormActivation(
                    out_channels, 3, kernel_size=4, stride=2, padding=1, conv_layer=nn.ConvTranspose2d,
                    norm_layer=None, act_layer=None
                ),
                nn.Tanh()
            ]
        )
        self.decoder = nn.Sequential(*layers)

    def encode_code(self, x):
        with torch.no_grad():
            # clip input to [-1, 1] range
            x = 2 * x - 1
            z = self.encoder(x)
            indices = self.codebook(z)[2]
            return indices

    def decode_code(self, latents):
        with torch.no_grad():
            latents = self.codebook.embeddings(latents).permute(0, 3, 1, 2).contiguous()
            return self.decoder(latents).permute(0, 2, 3, 1).cpu() * 0.5 + 0.5  # channels last format + mapped to [0,1]

    def forward(self, x):
        """
        Args:
            x: Original input of size (B, C, H, W). Note that x is expected to be in [-1, 1] range!

        Returns:
            Reconstructed version of input (decoder's output)
            Vector Quantization loss: L2 error between the embedding space and the encoder's outputs.
                Here, the stop-gradient operator is applied to the encoder's output since it is used
                to update the codebook.
            Commitment loss: encourage an encoder's output to stay close to the embedding space
                and prevent it from frequently fluctuating between latent codes. To do so, we
                apply the stop-gradient operator to the latent codes.
        """

        z = self.encoder(x)
        e, e_st, _ = self.codebook(z)
        x_tilde = self.decoder(e_st)

        # VQ loss
        vq_loss = torch.mean((z.detach() - e) ** 2)
        # Commitment loss
        commitment_loss = self.beta * torch.mean((e.detach() - z) ** 2)

        return x_tilde, vq_loss, commitment_loss

    def loss(self, x):
        x = 2 * x - 1
        x_tilde, vq_loss, commitment_loss = self(x)
        recon_loss = F.mse_loss(x_tilde, x)
        final_loss = recon_loss + vq_loss + commitment_loss

        return dict(loss=final_loss, recon_loss=recon_loss, vq_loss=vq_loss, commitment_loss=commitment_loss)


class VQVAE2(nn.Module):
    """
    Implements the hierarchical VQ-VAE model (https://arxiv.org/pdf/1906.00446.pdf).
    As in the original paper, we use a two-level latent hierarchy. The bottom level
    encoder downsamples the input by a factor of 4, whereas this output is further
    downsampled by a factor of 2 using the top level encoder.

    Note that all args are the same as in the case of the vanilla VQ-VAE model.
    """

    def __init__(self, out_channels=128, num_blocks=2, code_dim=128, code_size=128, beta=1.):
        super().__init__()
        self.beta = beta

        # (Bottom + top) encoder architecture
        layers = [
            Conv2dNormActivation(3, out_channels, kernel_size=4, stride=2, padding=1),
            Conv2dNormActivation(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ]
        for _ in range(num_blocks):
            layers.append(
                ResBlock(out_channels, out_channels)
            )
        self.bottom_encoder = nn.Sequential(*layers)

        layers = [
            Conv2dNormActivation(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ]
        for _ in range(num_blocks):
            layers.append(
                ResBlock(out_channels, out_channels)
            )
        self.top_encoder = nn.Sequential(*layers)

        # intermediate conv layers
        self.top_quantize_conv = None
        if out_channels != code_dim:
            self.top_quantize_conv = nn.Conv2d(out_channels, code_dim, 1, bias=False)
        self.bottom_quantize_conv = nn.Conv2d(out_channels + code_dim, code_dim, 1, bias=False)

        # (Bottom + top) codebooks
        self.bottom_codebook = Quantize(code_dim, code_size)
        self.top_codebook = Quantize(code_dim, code_size)

        # upsampling (for top-level latent codes)
        self.top_upsample_conv = nn.ConvTranspose2d(
            code_dim, code_dim, kernel_size=4, stride=2, padding=1, bias=False
        )

        # (Bottom + top) Decoder architecture
        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock(code_dim, code_dim))
        layers.append(
            Conv2dNormActivation(
                code_dim, code_dim, kernel_size=4, stride=2, padding=1, conv_layer=nn.ConvTranspose2d
            )
        )
        self.top_decoder = nn.Sequential(*layers)

        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock(code_dim * 2, code_dim * 2))
        layers.extend(
            [
                Conv2dNormActivation(
                    code_dim * 2, code_dim * 2, kernel_size=4, stride=2, padding=1, conv_layer=nn.ConvTranspose2d
                ),
                Conv2dNormActivation(
                    code_dim * 2, 3, kernel_size=4, stride=2, padding=1, conv_layer=nn.ConvTranspose2d,
                    norm_layer=None, act_layer=None
                ),
                nn.Tanh()
            ]
        )
        self.bottom_decoder = nn.Sequential(*layers)

    def encode_code(self, x):
        with torch.no_grad():
            b_enc = self.bottom_encoder(x)
            t_enc = self.top_encoder(b_enc)
            if self.top_quantize_conv is not None:
                t_enc = self.top_quantize_conv(t_enc)
            _, t_quant_st, t_indices = self.top_codebook(t_enc)

            b_enc = torch.cat((b_enc, self.top_decoder(t_quant_st)), dim=1)
            b_enc = self.bottom_quantize_conv(b_enc)
            _, _, b_indices = self.bottom_codebook(b_enc)

        return t_indices, b_indices  # top & bottom latent code indices

    def decode_code(self, z_top, z_bottom):
        with torch.no_grad():
            t_quant = self.top_codebook.embeddings(z_top).permute(0, 3, 1, 2)
            t_quant = self.top_upsample_conv(t_quant)
            b_quant = self.bottom_codebook.embeddings(z_bottom).permute(0, 3, 1, 2)
            quant = torch.cat((b_quant, t_quant), dim=1)

        return self.bottom_decoder(quant).permute(0, 2, 3, 1).cpu() * 0.5 + 0.5  # channels last + [0, 1] range

    def forward(self, x):
        # encode input
        b_enc = self.bottom_encoder(x)
        t_enc = self.top_encoder(b_enc)
        if self.top_quantize_conv is not None:
            t_enc = self.top_quantize_conv(t_enc)

        # top-level quantization
        t_quant, t_quant_st, _ = self.top_codebook(t_enc)

        # append to bottom-level feature maps
        b_enc = torch.cat((b_enc, self.top_decoder(t_quant_st)), dim=1)
        b_enc = self.bottom_quantize_conv(b_enc)

        # bottom-level quantization
        b_quant, b_quant_st, _ = self.bottom_codebook(b_enc)

        # final decoding
        quant_st = torch.cat((b_quant_st, self.top_upsample_conv(t_quant_st)), dim=1)
        dec = self.bottom_decoder(quant_st)

        # (top + bottom) VQ loss
        t_vq_loss = torch.mean((t_enc.detach() - t_quant) ** 2)
        b_vq_loss = torch.mean((b_enc.detach() - b_quant) ** 2)

        # (top + bottom) Commitment loss
        t_commitment_loss = self.beta * torch.mean((t_quant.detach() - t_enc) ** 2)
        b_commitment_loss = self.beta * torch.mean((b_quant.detach() - b_enc) ** 2)

        return dec, t_vq_loss + b_vq_loss, t_commitment_loss + b_commitment_loss

    def loss(self, x):
        x = 2 * x - 1
        x_tilde, vq_loss, commitment_loss = self(x)
        recon_loss = F.mse_loss(x_tilde, x)
        final_loss = recon_loss + vq_loss + commitment_loss

        return dict(loss=final_loss, recon_loss=recon_loss, vq_loss=vq_loss, commitment_loss=commitment_loss)
