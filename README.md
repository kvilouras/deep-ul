# deep-ul

Unofficial PyTorch implementation of the Vector-Quantized Variational AutoEncoder (VQ-VAE) architecture and its hierarchical variant (VQ-VAE-2).

A thorough analysis of our results on STL-10 dataset can be found [here](https://wandb.ai/kostasvl/image-vq-vae/reports/Vector-Quantized-Variational-Autoencoders-VQ-VAEs---VmlldzoyNDAxNjg0?accessToken=tvqs9vtp5vha5aqfms8db3y233na6bw24dn49glg5l6hmfwicblkppbwkmaa500l).

Tested on `torch>=1.7.1`, `torchvision>=0.8.2` and `wandb==0.12.21`.

## Vanilla VQ-VAE
To train a vanilla VQ-VAE model, run the following:
```python3
python3 train_vqvae.py configs/image_config.yaml
```
You can enable logging to wandb via the switch `--use-wandb`. 
To run the model on inference mode, use the `--test-only` switch. 
To load a checkpoint from a previous run that was stored in wandb, set an environment variable as follows: 
```python
os.environ["RUN_PATH"]='username/project/run-id'
```
## Gated PixelCNN prior
To train an autoregressive prior based on the Gated PixelCNN architecture, run the following:
```python3
python3 train_prior.py configs/prior_image_config.yaml configs/image_config.yaml
```
There are two additional switches here: `--class-conditional` enables class-conditional generation and `--num-classes` refers to the total number of classes available in the dataset (to convert labels to one-hot vectors). To download a pre-trained VQ-VAE model from wandb, set the following env variable:
```python
os.environ["VAE_RUN_PATH"]='username/project/run-id'
```

## Hierarchical VQ-VAE (VQ-VAE-2)
To train a VQ-VAE-2 model from scratch, run the following (the rest are the same as in the case of vanilla VQ-VAE):
```python3
python3 train_vqvae.py configs/vqvae2_config.yaml 
```

## Hierarchical PixelSNAIL prior
To train a hierarchical PixelSNAIL prior model (only compatible with VQ-VAE-2 model), run the following:
```python3
python3 train_pixelsnail.py configs/pixelsnail_prior_config.yaml configs/vqvae2_config.yaml
```
The rest are the same as in the case of Gated PixelCNN.
