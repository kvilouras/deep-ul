import datetime
from collections import deque


class Logger(object):
    def __init__(self, log_fn=None, prefix=""):
        self.log_fn = log_fn
        self.prefix = ""
        if prefix:
            self.prefix = prefix + ' | '

    def add_line(self, content):
        msg = self.prefix + content
        fp = open(self.log_fn, 'a')
        fp.write(msg + '\n')
        fp.flush()
        fp.close()


class ProgressMeter(object):
    def __init__(self, num_batches, meters, phase, epoch=None, logger=None):
        self.batches_per_epoch = num_batches
        self.batch_fmtst = self._get_batch_fmtstr(epoch, num_batches)
        self.meters = meters
        self.phase = phase
        self.epoch = epoch
        self.logger = logger

    def display(self, batch):
        date = str(datetime.datetime.now())
        entries = [f"{date} | {self.phase} {self.batch_fmtst.format(batch)}"]
        entries += [str(m) for m in self.meters]
        if self.logger is None:
            print('\t'.join(entries))
        else:
            self.logger.add_line('\t'.join(entries))

    def _get_batch_fmtstr(self, epoch, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        epoch_str = '[{}]'.format(epoch) if epoch is not None else ''
        return epoch_str + '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """
    Compute + store both the average and the current value
    of a given metric
    """
    def __init__(self, name, fmt=':f', window_size=0):
        self.name = name
        self.fmt = fmt
        self.window_size = window_size
        self.reset()

    def reset(self):
        if self.window_size > 0:
            self.q = deque(maxlen=self.window_size)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        if self.window_size > 0:
            self.q.append((val, n))
            self.count = sum([n for v, n in self.q])
            self.sum = sum([v * n for v, n in self.q])
        else:
            self.sum += val * n
            self.count += n

        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
