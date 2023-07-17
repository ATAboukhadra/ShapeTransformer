# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from __future__ import annotations

import os
import sys
import time

import torch
import torch.backends.cudnn


class Logger:
    __slots__ = "log", "start_line"

    def __init__(self, log_dir: str):
        """Create a logger."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log = open(os.path.join(log_dir, "log.md"), "w")

        self.log.write("# Environment  \n")
        self.log.write(f"|Parameter|Value|\n")
        self.log.write(f"|---------|-----|\n")
        self.log.write(f"|pyTorch|{torch.__version__}|\n")
        self.log.write(f"|cuDNN|{torch.backends.cudnn.version()}|\n")
        self.log.write(f"|GPU count|{torch.cuda.device_count()}|\n")
        message = f"|cmd line|"
        for item in sys.argv:
            message += " " + str(item)
        message += "|\n"
        self.log.write(message)
        self.log.write("\n# Log Messages  \n")
        self.start_line = True

    def write(self, txt: str):
        if self.start_line:
            time_str = time.strftime("%Y-%m-%d-%H-%M")
            self.log.write(f"{time_str}: {txt}")
        else:
            self.log.write(txt)
        self.start_line = False
        if "\n" in txt:
            self.start_line = True

    def flush(self):
        self.log.flush()

    def close(self):
        self.log.close()
