import os
import datetime
import shutil

import torch
import torch.utils.data as data
from torchvision.datasets import STL10
from torchvision.transforms import Resize, ToTensor, Compose

from .logger import Logger


def prep_env(args, cfg):
    model_dir = '{}/{}'.format(cfg['model']['model_dir'], cfg['model']['name'])
    os.makedirs(model_dir, exist_ok=True)
    log_fn = '{}/train.log'.format(model_dir)
    logger = Logger(log_fn)

    logger.add_line(str(datetime.datetime.now()))
    logger.add_line('=' * 30 + ' Config ' + '=' * 30)

    def print_dict(d, ident=''):
        for k in d:
            if isinstance(d[k], dict):
                logger.add_line("{}{}".format(ident, k))
                print_dict(d[k], ident='  ' + ident)
            else:
                logger.add_line("{}{}: {}".format(ident, k, str(d[k])))

    print_dict(cfg)
    logger.add_line('=' * 30 + ' Args ' + '=' * 30)
    for k in args.__dict__:
        logger.add_line('{:30} {}'.format(k, args.__dict__[k]))

    return logger, model_dir


def build_dataloaders(cfg):
    if cfg['name'] == 'stl10':
        directory = cfg['data_dir'] if 'data_dir' in cfg else os.getcwd()
        resize = (cfg['resize'], cfg['resize'])
        train_ds = STL10(
            root=directory,
            split="train",
            transform=Compose(
                [
                    Resize(resize),
                    ToTensor()
                ]
            ),
            download=True
        )

        test_ds = STL10(
            root=directory,
            split="test",
            transform=Compose(
                [
                    Resize(resize),
                    ToTensor()
                ]
            ),
            download=True
        )
    else:
        raise ValueError("Unknown dataset name")

    train_dl = data.DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True
    )

    test_dl = data.DataLoader(
        test_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True
    )

    return train_dl, test_dl


def build_model(cfg, logger=None):
    import models
    assert cfg['arch'] in models.__dict__, 'Unknown network architecture'
    model = models.__dict__[cfg['arch']](**cfg['args'])

    if logger is not None:
        if isinstance(model, (list, tuple)):
            logger.add_line('=' * 30 + ' Model ' + '=' * 30)
            for m in model:
                logger.add_line(str(m))
            logger.add_line('=' * 30 + ' Parameters ' + '=' * 30)
            for m in model:
                logger.add_line(param_description(m))
        else:
            logger.add_line('=' * 30 + ' Model ' + '=' * 30)
            logger.add_line(str(model))
            logger.add_line('=' * 30 + ' Parameters ' + '=' * 30)
            logger.add_line(param_description(model))

    if torch.cuda.is_available():
        if isinstance(model, (list, tuple)):
            for i in range(len(model)):
                model[i] = model[i].cuda()
        else:
            model = model.cuda()

    return model


def param_description(model):
    desc = ''
    for n, p in model.named_parameters():
        desc += '{:70} | {:10} | {:30} | {}\n'.format(
            n, 'Trainable' if p.requires_grad else 'Frozen',
            ' x '.join([str(s) for s in p.size()]),
            str(p.numel())
        )

    return desc


class CheckpointManager(object):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = 0.  # TODO: possibly a large number instead of 0?

    def save(self, epoch, filename=None, eval_metric=0., **kwargs):
        is_best = False
        if self.best_metric < eval_metric:
            is_best = True
            self.best_metric = eval_metric

        state = dict(epoch=epoch)
        for k in kwargs:
            state[k] = kwargs[k].state_dict()

        if filename:
            self.save_checkpoint(state, is_best=False, filename='{}/{}'.format(self.checkpoint_dir, filename))
        else:
            self.save_checkpoint(state, is_best, model_dir=self.checkpoint_dir)

    def save_checkpoint(self, state, is_best, model_dir='.', filename=None):
        if filename is None:
            filename = '{}/checkpoint.pth.tar'.format(model_dir)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, '{}/model_best.pth.tar'.format(model_dir))

    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def custom_checkpoint_fn(self, filename):
        return '{}/{}'.format(self.checkpoint_dir, filename)

    def checkpoint_fn(self, last=False, best=False):
        assert last or best
        if last:
            return self.last_checkpoint_fn()
        else:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def custom_checkpoint_exists(self, filename):
        return os.path.isfile(self.custom_checkpoint_fn(filename))

    def restore(self, fn=None, restore_last=False, restore_best=False, **kwargs):
        checkpoint_fn = fn if fn else self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        start_epoch = ckp['epoch']
        for k in kwargs:
            kwargs[k].load_state_dict(ckp[k])

        return start_epoch
