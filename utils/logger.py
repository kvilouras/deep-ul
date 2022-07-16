import sys


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
