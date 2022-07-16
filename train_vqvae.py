import argparse
import shutil
import os
import yaml
from tqdm import tqdm
import numpy as np

import torch.nn as nn
from torch.backends import cudnn
import torch.optim

from utils import train_utils
from utils.vis import visualize_data


parser = argparse.ArgumentParser(description='VQ-VAE Training')

parser.add_argument('cfg', help='Path to config file')
parser.add_argument('--test-only', action='store_true', help='Run VQ-VAE model on inference mode')
parser.add_argument('--use-wandb', action='store_true', help='Enable WandB logging')


def main():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    use_gpu = torch.cuda.is_available()

    # prepare wandb (optional). In this case, some info should be in the input config file.
    if args.use_wandb:
        import wandb
        run = wandb.init(project=cfg['wandb']['project_name'], config=cfg, resume=cfg['wandb']['resume'])

    # setup environment
    logger, model_dir = train_utils.prep_env(args, cfg)

    # instantiate datasets + loaders
    train_dl, test_dl = train_utils.build_dataloaders(cfg['dataset'])

    # instantiate model
    model = train_utils.build_model(cfg['model'])

    # instantiate optimizer
    optimizer = torch.optim.Adam(
        params=list(model.parameters()),
        lr=cfg['optimizer']['lr'],
        weight_decay=cfg['optimizer']['weight_decay'] if 'weight_decay' in cfg['optimizer'] else 0
    )

    # log train/test samples
    if args.use_wandb:
        grid = visualize_data(train_dl.dataset, num_imgs=100, nrow=10, return_grid=True).permute(1, 2, 0).numpy()
        img = wandb.Image(grid, caption='Train data')
        wandb.log({'training_examples': img})
        grid = visualize_data(test_dl.dataset, num_imgs=100, nrow=10, return_grid=True).permute(1, 2, 0).numpy()
        img = wandb.Image(grid, caption='Test data')
        wandb.log({'test_examples': img})
        wandb.watch(model)  # also log gradients of weights

    # TODO: add a lr scheduler if needed

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
            wandb.restore('checkpoint.pth.tar', run_path=os.environ['RUN_PATH'])
            shutil.move('checkpoint.pth.tar', ckp_manager.last_checkpoint_fn())
            start_epoch = ckp_manager.restore(restore_last=True, model=model, optimizer=optimizer)
            logger.add_line("Checkpoint loaded from WandB'{}' (epoch {})".format(
                ckp_manager.last_checkpoint_fn(), start_epoch
            ))
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
                    # log reconstructions to wandb
                    recon = reconstruct_samples(test_dl, model, use_gpu, n=25)
                    wandb.log({f'recon_ep{epoch}': wandb.Image(
                        recon.numpy(), caption=f'Reconstructions Epoch {epoch}'
                    )})

            if args.use_wandb:
                # log losses per epoch
                wandb.log(
                    dict(
                        **{'train/' + k: v for k, v in train_losses.items()},
                        **{'test/' + k: v for k, v in test_losses.items()},
                        epoch=epoch
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
        # log reconstructions to wandb
        if args.use_wandb:
            recon = reconstruct_samples(test_dl, model, use_gpu, n=25)
            wandb.log({'final_reconstructions': wandb.Image(recon.numpy(), caption='Final reconstructions')})

    if args.use_wandb:
        # save final model to wandb
        wandb.save(ckp_manager.last_checkpoint_fn())
        run.finish()


def run_phase(phase, loader, model, optimizer, epoch, cfg, logger, use_gpu):
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))

    model.train(phase == 'train')
    pbar = tqdm(total=len(loader.dataset))  # progress bar
    losses = dict()
    for i, sample in enumerate(loader):
        x = sample[0]  # discard label here
        if use_gpu:
            x = x.cuda(non_blocking=True)

        if phase == 'train':
            out = model.loss(x)
            optimizer.zero_grad()
            out['loss'].backward()
            if cfg['grad_clip']:
                nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])  # gradient clipping (optional)
            optimizer.step()
            desc = f"Epoch {epoch}"
            for k, v in out.items():
                if k not in losses:
                    losses[k] = []
                losses[k].append(v.item())
                avg_loss = np.mean(losses[k][-50:])
                desc += f", {k} {avg_loss:.4f}"
                pbar.set_description(desc)
                pbar.update(x.shape[0])
        else:
            with torch.no_grad():
                out = model.loss(x)
                for k, v in out.items():
                    losses[k] = losses.get(k, 0) + v.item() * x.shape[0]

    pbar.close()

    if phase != 'train':
        desc = "Test "
        for k in losses.keys():
            losses[k] /= len(loader.dataset)
            desc += f", {k} {losses[k]:.4f}"
        logger.add_line(desc)

    return losses


def reconstruct_samples(loader, model, use_gpu, n=25):
    """
    Return n pairs of original/reconstructed samples
    Note that n mut be <= mini-batch size.
    """
    x = next(iter(loader))[:n]
    _, c, h, w = x.shape
    if use_gpu:
        x = x.cuda()
    with torch.no_grad():
        z = model.encode_code(x)
        x_recon = model.decode_code(z)
    x = x.cpu().permute(0, 2, 3, 1)
    reconstructions = torch.stack((x, x_recon), dim=1).reshape((-1, h, w, c))

    return reconstructions


if __name__ == '__main__':
    main()