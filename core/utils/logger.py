import os
import sys


class Logger(object):
    def __init__(self, fpath):
        self.console = sys.stdout
        self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
        else:
            self.console.write("Warning: Log file is None")

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
        else:
            self.console.write("Warning: Log file is None")

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
