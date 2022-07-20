import argparse
import os
import yaml
import numpy as np

from torch.utils import data
import torch.nn as nn
from torch.backends import cudnn
import torch.optim
from torchvision.utils import make_grid

from utils import train_utils
from utils.logger import ProgressMeter, AverageMeter
from utils.vis import visualize_data


parser = argparse.ArgumentParser(description='Prior (Gated PixelCNN) Training')

parser.add_argument('cfg', help='Path to config file')
parser.add_argument('vae_cfg', help='Path to VQ-VAE model config file')
parser.add_argument('--test-only', action='store_true', help='Run prior model on inference mode')
parser.add_argument('--use-wandb', action='store_true', help='Enable WandB logging')


def main():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    vae_cfg = yaml.safe_load(open(args.vae_cfg))
    use_gpu = torch.cuda.is_available()

    # prepare wandb (optional). In this case, some info should be in the input config file.
    if args.use_wandb:
        import wandb
        run = wandb.init(project=cfg['wandb']['project_name'], config=cfg, resume=cfg['wandb']['resume'])

    # setup environment
    logger, model_dir = train_utils.prep_env(args, cfg)

    # instantiate datasets + loaders
    train_dl, test_dl = train_utils.build_dataloaders(cfg['dataset'])

    # log train/test samples
    if args.use_wandb:
        grid = visualize_data(train_dl.dataset, num_imgs=100, nrow=10, return_grid=True).permute(1, 2, 0).numpy()
        img = wandb.Image(grid, caption='Train data')
        wandb.log({'training_examples': img})
        grid = visualize_data(test_dl.dataset, num_imgs=100, nrow=10, return_grid=True).permute(1, 2, 0).numpy()
        img = wandb.Image(grid, caption='Test data')
        wandb.log({'test_examples': img})

    # instantiate (pre-trained) VQ-VAE model
    vae_model = train_utils.build_model(vae_cfg['model'], logger=logger)
    vae_model_dir = '{}/{}/{}'.format(vae_cfg['model']['model_dir'], vae_cfg['model']['name'], 'checkpoint.pth.tar')
    if os.path.isfile(vae_model_dir):
        ckp = torch.load(vae_model_dir, map_location={'cuda:0': 'cpu'})
        vae_epoch = ckp['epoch']
        vae_model.load_state_dict(ckp['model'])
        logger.add_line("VAE checkpoint loaded '{}' (epoch {})".format(vae_model_dir, vae_epoch))
    elif args.use_wandb:
        # restore VAE model checkpoint from wandb
        # (run_path is expected to be an env variable of type 'username/project/run-id')
        try:
            wandb.restore(
                vae_model_dir, run_path=os.environ['VAE_RUN_PATH'], replace=False, root=os.getcwd()
            )
            ckp = torch.load(vae_model_dir, map_location={'cuda:0': 'cpu'})
            vae_epoch = ckp['epoch']
            vae_model.load_state_dict(ckp['model'])
            logger.add_line("VAE checkpoint loaded from WandB'{}' (epoch {})".format(
                vae_model_dir, vae_epoch
            ))
        except (ValueError, KeyError, wandb.errors.CommError):
            logger.add_line("No VAE checkpoint found in {}".format(vae_model_dir))
    else:
        logger.add_line("No VAE checkpoint found in {}".format(vae_model_dir))

    # Create prior dataset, i.e. map input images to latent code indices,
    # which will then be fed to Gated PixelCNN
    prior_train_ds, input_shape = create_prior_dataset(vae_model, train_dl, use_gpu)
    train_dl = data.DataLoader(
        prior_train_ds, batch_size=cfg['dataset']['batch_size'], num_workers=cfg['dataset']['num_workers'],
        pin_memory=True, shuffle=True
    )
    cfg['model']['args']['input_shape'] = input_shape
    prior_test_ds, _ = create_prior_dataset(vae_model, test_dl, use_gpu)
    test_dl = data.DataLoader(
        prior_test_ds, batch_size=cfg['dataset']['batch_size'], num_workers=cfg['dataset']['num_workers'],
        pin_memory=True, shuffle=False
    )

    # instantiate prior model
    model = train_utils.build_model(cfg['model'], logger=logger)

    # instantiate optimizer
    optimizer = torch.optim.Adam(
        params=list(model.parameters()),
        lr=cfg['optimizer']['lr'],
        weight_decay=cfg['optimizer']['weight_decay'] if 'weight_decay' in cfg['optimizer'] else 0
    )

    if args.use_wandb:
        wandb.watch(model)  # also log gradients of weights

    # checkpoint manager
    ckp_manager = train_utils.CheckpointManager(model_dir)

    # optionally resume from a checkpoint
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    if cfg['resume']:
        if ckp_manager.checkpoint_exists(last=True):
            start_epoch = ckp_manager.restore(restore_last=True, model=model, optimizer=optimizer)
            logger.add_line("Checkpoint loaded '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch))
        elif args.use_wandb:
            # restore model checkpoint from wandb
            # (run_path is expected to be an env variable of type 'username/project/run-id')
            try:
                wandb.restore(
                    ckp_manager.last_checkpoint_fn(), run_path=os.environ['PRIOR_RUN_PATH'],
                    replace=False, root=os.getcwd()
                )
                start_epoch = ckp_manager.restore(restore_last=True, model=model, optimizer=optimizer)
                logger.add_line("Checkpoint loaded from WandB'{}' (epoch {})".format(
                    ckp_manager.last_checkpoint_fn(), start_epoch
                ))
            except (ValueError, KeyError, wandb.errors.CommError):
                logger.add_line("No checkpoint found in {}".format(ckp_manager.last_checkpoint_fn()))
        else:
            logger.add_line("No checkpoint found in {}".format(ckp_manager.last_checkpoint_fn()))

    cudnn.benchmark = True

    if not args.test_only:
        # Training phase
        test_freq = cfg['test_freq'] if 'test_freq' in cfg else 1
        train_losses, test_losses = dict(), dict()
        for epoch in range(start_epoch, end_epoch):
            train_loss = run_phase('train', train_dl, model, optimizer, epoch, cfg['optimizer'], logger, use_gpu)
            test_loss = run_phase('test', test_dl, model, optimizer, epoch, cfg['optimizer'], logger, use_gpu)

            for k in train_loss.keys():
                if k not in train_losses:
                    train_losses[k] = []
                    test_losses[k] = []
                train_losses[k].extend(train_loss[k])
                test_losses[k].append(test_loss[k])

            if epoch % test_freq == 0 or epoch == end_epoch - 1:
                ckp_manager.save(epoch + 1, model=model, optimizer=optimizer)
                # save model checkpoint to wandb every 5 epochs
                if args.use_wandb and epoch % 5 == 0 and epoch != 0 and epoch != end_epoch - 1:
                    wandb.save(ckp_manager.last_checkpoint_fn(), policy="now")
                    # log generated samples to wandb
                    gen = generate_samples(model, vae_model, n=50)
                    wandb.log({f'gen_ep{epoch}': wandb.Image(
                        gen.numpy(), caption=f'Generated samples Epoch {epoch}'
                    )})

            if args.use_wandb:
                # log losses per epoch
                wandb.log(
                    dict(
                        **{'train/' + k: np.mean(v[-50:]) for k, v in train_losses.items()},
                        **{'test/' + k: np.mean(v) for k, v in test_losses.items()},
                        step=epoch
                    )
                )
    else:
        # Inference mode
        test_losses = dict()
        test_loss = run_phase('test', test_dl, model, optimizer, end_epoch, cfg['optimizer'], logger, use_gpu)
        for k in test_loss.keys():
            if k not in test_losses:
                test_losses[k] = []
            test_losses[k].append(test_loss[k])
        # log losses + generated samples to wandb
        if args.use_wandb:
            wandb.log(
                dict(
                    **{'inference/' + k: np.mean(v) for k, v in test_losses.items()}
                )
            )
            gen = generate_samples(model, vae_model, n=50)
            wandb.log({'final_gen': wandb.Image(
                gen.numpy(), caption='Final generated samples')}
            )

    if args.use_wandb:
        # save final model to wandb
        wandb.save(ckp_manager.last_checkpoint_fn())
        run.finish()


def create_prior_dataset(model, loader, use_gpu):
    model.train(False)
    prior_data = []
    for i, sample in enumerate(loader):
        x, _ = sample
        if use_gpu:
            x = x.cuda(non_blocking=True)
        z = model.encode_code(x)  # indices
        if i == 0:
            input_shape = list(z.shape[1:])
        prior_data.append(z.cpu().long())

    prior_data = torch.cat(prior_data, dim=0)

    return prior_data, input_shape


def run_phase(phase, loader, model, optimizer, epoch, cfg, logger, use_gpu):
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    nll_loss_meter = AverageMeter('NLL Loss', ':.4f')
    bpd_meter = AverageMeter('Bits/dim', ':.4f')
    progress = ProgressMeter(
        len(loader), [nll_loss_meter, bpd_meter], phase=phase, epoch=epoch, logger=logger
    )

    model.train(phase == 'train')
    losses = dict()
    for i, sample in enumerate(loader):
        x = sample
        if use_gpu:
            x = x.cuda(non_blocking=True)

        if phase == 'train':
            out = model.loss(x)
            optimizer.zero_grad()
            out['nll_loss'].backward()
            if cfg['grad_clip']:
                nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])  # gradient clipping (optional)
            optimizer.step()
            for k, v in out.items():
                if k not in losses:
                    losses[k] = []
                losses[k].append(v.item())
            # update meters
            nll_loss_meter.update(out['nll_loss'].item(), x.shape[0])
            bpd_meter.update(out['bpd'].item(), x.shape[0])
            # show progress
            if (i + 1) % 100 == 0 or i == 0 or i == len(loader) - 1:
                progress.display(i + 1)
        else:
            with torch.no_grad():
                out = model.loss(x)
                for k, v in out.items():
                    losses[k] = losses.get(k, 0) + v.item() * x.shape[0]

    if phase != 'train':
        desc = "Test"
        for k in losses.keys():
            losses[k] /= len(loader.dataset)
            desc += f", {k} {losses[k]:.4f}"
        logger.add_line(desc)

    return losses


def generate_samples(prior_model, vae_model, n=100):
    """
    Return n randomly generated samples
    """

    samples = prior_model.sample(n).long()
    x_gen = vae_model.decode_code(samples).permute(0, 3, 1, 2).contiguous()
    x_gen = make_grid(x_gen, nrow=10).permute(1, 2, 0)

    return x_gen


if __name__ == '__main__':
    main()
