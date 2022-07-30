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


parser = argparse.ArgumentParser(description='Hierarchical Prior (PixelSNAIL) Training')

parser.add_argument('cfg', help='Path to config file')
parser.add_argument('vae_cfg', help='Path to VQ-VAE model config file')
parser.add_argument('--test-only', action='store_true', help='Run prior model on inference mode')
parser.add_argument('--use-wandb', action='store_true', help='Enable WandB logging')
parser.add_argument('--class-conditional', action='store_true', help='Enable class-conditional generation')
parser.add_argument('--num-classes', type=int, default=None, help='Total number of classes (for '
                                                                  'class-conditional generation)')


def main():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    vae_cfg = yaml.safe_load(open(args.vae_cfg))
    use_gpu = torch.cuda.is_available()
    if args.class_conditional:
        assert args.num_classes, "Number of classes is not given"

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

    # Create prior datasets, i.e. map input images to latent code indices
    prior_train_ds, input_shape = create_prior_dataset(vae_model, train_dl, args, use_gpu)
    top_prior_train_ds = prior_train_ds[0]
    if args.class_conditional:
        top_prior_train_ds = data.TensorDataset(prior_train_ds[0], prior_train_ds[-1])
        cfg['model']['args']['conditional_size'] = args.num_classes
    cfg['model']['args']['top_input_shape'] = input_shape[0]
    bottom_prior_train_ds = data.TensorDataset(*prior_train_ds)
    cfg['model']['args']['bottom_input_shape'] = input_shape[1]
    top_train_dl = data.DataLoader(
        top_prior_train_ds, batch_size=cfg['dataset']['batch_size'], num_workers=cfg['dataset']['num_workers'],
        pin_memory=True, shuffle=True
    )
    bottom_train_dl = data.DataLoader(
        bottom_prior_train_ds, batch_size=cfg['dataset']['batch_size'], num_workers=cfg['dataset']['num_workers'],
        pin_memory=True, shuffle=True
    )
    train_dl = [top_train_dl, bottom_train_dl]

    prior_test_ds, _ = create_prior_dataset(vae_model, test_dl, args, use_gpu)
    top_prior_test_ds = prior_test_ds[0]
    if args.class_conditional:
        top_prior_test_ds = data.TensorDataset(prior_test_ds[0], prior_test_ds[-1])
    bottom_prior_test_ds = data.TensorDataset(*prior_test_ds)
    top_test_dl = data.DataLoader(
        top_prior_test_ds, batch_size=cfg['dataset']['batch_size'], num_workers=cfg['dataset']['num_workers'],
        pin_memory=True, shuffle=False
    )
    bottom_test_dl = data.DataLoader(
        bottom_prior_test_ds, batch_size=cfg['dataset']['batch_size'], num_workers=cfg['dataset']['num_workers'],
        pin_memory=True, shuffle=False
    )
    test_dl = [top_test_dl, bottom_test_dl]

    # instantiate (top & bottom) prior models
    models = train_utils.build_model(cfg['model'], logger=logger)

    # instantiate optimizers
    top_optimizer = torch.optim.Adam(
        params=list(models[0].parameters()),
        lr=cfg['optimizer']['lr'],
        weight_decay=cfg['optimizer']['weight_decay'] if 'weight_decay' in cfg['optimizer'] else 0
    )
    bottom_optimizer = torch.optim.Adam(
        params=list(models[1].parameters()),
        lr=cfg['optimizer']['lr'],
        weight_decay=cfg['optimizer']['weight_decay'] if 'weight_decay' in cfg['optimizer'] else 0
    )
    optimizers = [top_optimizer, bottom_optimizer]

    if args.use_wandb:
        wandb.watch(models[0])  # also log gradients of weights
        wandb.watch(models[1])

    # checkpoint manager
    ckp_manager = train_utils.CheckpointManager(model_dir)

    # optionally resume from a checkpoint
    start_epochs, end_epoch = [0, 0], cfg['optimizer']['num_epochs']
    if cfg['resume']:
        for i, ext in enumerate(['top', 'bottom']):
            fn = 'checkpoint_{}.pth.tar'.format(ext)
            if ckp_manager.custom_checkpoint_exists(fn):
                start_epoch = ckp_manager.restore(fn=fn, model=models[i], optimizer=optimizers[i])
                start_epochs[i] = start_epoch
                logger.add_line("{} checkpoint loaded {} (epoch {})".format(
                    ext, ckp_manager.custom_checkpoint_fn(fn), start_epoch)
                )
            elif args.use_wandb:
                # restore model checkpoint from wandb
                # (run_path is expected to be an env variable of type 'username/project/run-id')
                try:
                    wandb.restore(
                        ckp_manager.custom_checkpoint_fn(fn), run_path=os.environ[f'{ext.upper()}_PRIOR_RUN_PATH'],
                        replace=False, root=os.getcwd()
                    )
                    start_epoch = ckp_manager.restore(fn=fn, model=models[i], optimizer=optimizers[i])
                    start_epochs[i] = start_epoch
                    logger.add_line("{} checkpoint loaded from WandB'{}' (epoch {})".format(
                        ext, ckp_manager.custom_checkpoint_fn(fn), start_epoch
                    ))
                except (ValueError, KeyError, wandb.errors.CommError):
                    logger.add_line("No {} checkpoint found in {}".format(ext, ckp_manager.custom_checkpoint_fn(fn)))
            else:
                logger.add_line("No {} checkpoint found in {}".format(ext, ckp_manager.custom_checkpoint_fn(fn)))

    cudnn.benchmark = True

    if not args.test_only:
        # Training phase
        test_freq = cfg['test_freq'] if 'test_freq' in cfg else 1
        for i, ext in enumerate(['top', 'bottom']):
            fn = f"checkpoint_{ext}.pth.tar"
            train_losses, test_losses = dict(), dict()
            for epoch in range(start_epochs[i], end_epoch):
                train_loss = run_phase(
                    ext, 'train', train_dl[i], models[i], optimizers[i], epoch, cfg['optimizer'], args, logger, use_gpu
                )
                test_loss = run_phase(
                    ext, 'test', test_dl[i], models[i], optimizers[i], epoch, cfg['optimizer'], args, logger, use_gpu
                )

                for k in train_loss.keys():
                    if k not in train_losses:
                        train_losses[k] = []
                        test_losses[k] = []
                    train_losses[k].extend(train_loss[k])
                    test_losses[k].append(test_loss[k])

                if epoch % test_freq == 0 or epoch == end_epoch - 1:
                    ckp_manager.save(
                        epoch + 1, filename=fn, model=models[i], optimizer=optimizers[i]
                    )
                    # save model checkpoint to wandb every 5 epochs
                    if args.use_wandb and epoch % 5 == 0 and epoch != 0 and epoch != end_epoch - 1:
                        wandb.save(ckp_manager.custom_checkpoint_fn(fn), policy="now")
                        # log generated samples to wandb
                        if ext == 'bottom':
                            gen = generate_samples(models, vae_model, args, n=50)
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
        test_loss = run_phase(
            'bottom', 'test', test_dl[1], models[1], optimizers[1], end_epoch, cfg['optimizer'], args, logger, use_gpu
        )
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
            gen = generate_samples(models, vae_model, args, n=50)
            wandb.log({'final_gen': wandb.Image(
                gen.numpy(), caption='Final generated samples')}
            )

    if args.use_wandb:
        # save final model to wandb
        wandb.save(ckp_manager.custom_checkpoint_fn('checkpoint_top.pth.tar'))
        wandb.save(ckp_manager.custom_checkpoint_fn('checkpoint_bottom.pth.tar'))
        run.finish()


def create_prior_dataset(model, loader, args, use_gpu):
    model.train(False)
    top_prior_data, bottom_prior_data, prior_labels = [], [], []
    for i, sample in enumerate(loader):
        x, y = sample
        if use_gpu:
            x = x.cuda(non_blocking=True)
        z_top, z_bottom = model.encode_code(x)  # top & bottom level indices
        if i == 0:
            input_shape = (list(z_top.shape[1:]), list(z_bottom.shape[1:]))
        top_prior_data.append(z_top.cpu().long())
        bottom_prior_data.append(z_bottom.cpu().long())
        if args.class_conditional:
            # convert labels to one-hot (for unlabeled data, y is equal to -1)
            y_onehot = torch.zeros((y.shape[0], args.num_classes), dtype=torch.long)
            y_onehot[y != -1].scatter_(1, y[y != -1].long().unsqueeze(1), 1)
            prior_labels.append(y_onehot)

    top_prior_data = torch.cat(top_prior_data, dim=0)
    bottom_prior_data = torch.cat(bottom_prior_data, dim=0)
    if args.class_conditional:
        prior_labels = torch.cat(prior_labels, dim=0)

        return (top_prior_data, bottom_prior_data, prior_labels), input_shape

    return (top_prior_data, bottom_prior_data), input_shape


def run_phase(level, phase, loader, model, optimizer, epoch, cfg, args, logger, use_gpu):
    logger.add_line('\n{} level {}: Epoch {}'.format(level, phase, epoch))
    nll_loss_meter = AverageMeter('NLL Loss', ':.4f')
    bpd_meter = AverageMeter('Bits/dim', ':.4f')
    progress = ProgressMeter(
        len(loader), [nll_loss_meter, bpd_meter], phase=phase, epoch=epoch, logger=logger
    )

    model.train(phase == 'train')
    losses = dict()
    for i, sample in enumerate(loader):
        x_bottom, y = None, None
        if level == 'top':
            if args.class_conditional:
                x_top, y = sample
                y = y.float()
            else:
                x_top = sample
        else:
            if args.class_conditional:
                x_top, x_bottom, y = sample
                y = y.float()
            else:
                x_top, x_bottom = sample
        if use_gpu:
            x_top = x_top.cuda(non_blocking=True)
            if x_bottom is not None:
                x_bottom = x_bottom.cuda(non_blocking=True)
            if args.class_conditional:
                y = y.cuda(non_blocking=True)

        if phase == 'train':
            if level == 'top':
                out = model.loss(x_top, cond=y)
            else:
                out = model.loss(x_bottom, cond=y, top_condition=x_top)
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
            nll_loss_meter.update(out['nll_loss'].item(), x_top.shape[0])
            bpd_meter.update(out['bpd'].item(), x_top.shape[0])
            # show progress
            if (i + 1) % 100 == 0 or i == 0 or i == len(loader) - 1:
                progress.display(i + 1)
        else:
            with torch.no_grad():
                if level == 'top':
                    out = model.loss(x_top, cond=y)
                else:
                    out = model.loss(x_bottom, cond=y, top_condition=x_top)
                for k, v in out.items():
                    losses[k] = losses.get(k, 0) + v.item() * x_top.shape[0]

    if phase != 'train':
        desc = "Test"
        for k in losses.keys():
            losses[k] /= len(loader.dataset)
            desc += f", {k} {losses[k]:.4f}"
        logger.add_line(desc)

    return losses


def generate_samples(prior_models, vae_model, args, n=100):
    """
    Return n randomly generated samples
    """

    y_onehot = None
    if args.class_conditional:
        # generate labels + one-hot encoding
        y = torch.arange(args.num_classes, dtype=torch.long).repeat(n // args.num_classes).unsqueeze(1)
        y_onehot = torch.zeros((y.shape[0], args.num_classes), dtype=torch.float32)
        y_onehot.scatter_(1, y, 1)
    top_samples = prior_models[0].sample(n, cond=y_onehot).long()
    bottom_samples = prior_models[1].sample(n, cond=y_onehot, top_cond=top_samples).long()
    x_gen = vae_model.decode_code(top_samples, bottom_samples).permute(0, 3, 1, 2).contiguous()
    x_gen = make_grid(x_gen, nrow=10).permute(1, 2, 0)

    return x_gen


if __name__ == '__main__':
    main()
