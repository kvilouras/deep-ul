import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv2dNormActivation, ResBlock


__all__ = [
    'VQVAE',
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

