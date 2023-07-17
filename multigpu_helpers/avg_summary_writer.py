from __future__ import annotations

from typing import Any

import numpy as np
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class LogEntry:
    __slots__ = "value", "count", "global_step"

    def __init__(self, value: float | list[float], global_step: int):
        self.value = value
        self.count = 1
        self.global_step = global_step


class AvgSummaryWriter(SummaryWriter):
    """
    A wrapper class for the SummaryWriter.

    It averages all measurements it sees for a specific tag within the horizon of the specified sampling rate.
    The averaged values are then written to disk.
    """
    def __init__(self, log_dir=None, comment="", purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix="", sampling_rate: int = 10, histogram_sampling_rate: int = 200):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self.sampling_rate = sampling_rate
        self.hist_sampling_rate = histogram_sampling_rate
        self.value_dict: dict[str, LogEntry] = dict()

    def add_scalar(self, tag: str, scalar_value, global_step: int | None = None, walltime: float | None = None,
                   new_style: bool = False, double_precision: bool = False):
        if tag in self.value_dict and (global_step >= self.value_dict[tag].global_step + self.sampling_rate):
            # We have moved on to the next time step, so we write the averaged values into the log.
            entry = self.value_dict[tag]
            super().add_scalar(tag, entry.value / entry.count, entry.global_step, walltime, new_style, double_precision)
            del self.value_dict[tag]
        if isinstance(scalar_value, Tensor):
            scalar_value = float(scalar_value.detach().cpu().numpy())
        if tag not in self.value_dict:
            self.value_dict[tag] = LogEntry(scalar_value, global_step)
        else:
            entry = self.value_dict[tag]
            entry.value += scalar_value
            entry.count += 1

    def add_image(self, tag: str, img_tensor: Tensor, global_step: int | None = None,
                  walltime: float | None = None, dataformats: str = "CHW"):
        if (global_step + 1) % self.sampling_rate == 0:
            super().add_image(tag, img_tensor, global_step, walltime, dataformats)

    def add_images(self, tag: str, img_tensor: Tensor, global_step: int | None = None,
                   walltime: float | None = None, dataformats: str = "NCHW"):
        if (global_step + 1) % self.sampling_rate == 0:
            super().add_images(tag, img_tensor, global_step, walltime, dataformats)

    def add_histogram(self, tag: str, values: Tensor | list[Tensor], global_step: int | None = None,
                      bins: str = "tensorflow", walltime: float | None = None, max_bins: int | None = None):
        if tag in self.value_dict and (global_step >= self.value_dict[tag].global_step + self.hist_sampling_rate):
            entry = self.value_dict[tag]
            super().add_histogram(tag, np.asarray(entry.value), entry.global_step,
                                  bins=bins, walltime=walltime, max_bins=max_bins)
            del self.value_dict[tag]
        if isinstance(values, Tensor):
            values = float(values.detach().cpu().numpy().tolist())
        if tag not in self.value_dict:
            if isinstance(values, list):
                self.value_dict[tag] = LogEntry(values, global_step)
            else:
                self.value_dict[tag] = LogEntry([values], global_step)
        else:
            entry = self.value_dict[tag]
            if isinstance(values, list):
                entry.value.extend(values)
            else:
                entry.value.append(values)
            entry.count += 1

    def add_graph(self, model, input_to_model: Tensor | list[Tensor] | None = None,
                  verbose: bool = False, use_strict_trace: bool = True):
        super().add_graph(model, input_to_model, verbose, use_strict_trace)

    def add_hparams(self, hparam_dict: dict[str, Any], metric_dict: dict[str, Any],
                    hparam_domain_discrete: dict[str, list[Any]] = None, run_name: str | None = None):
        super().add_hparams(hparam_dict, metric_dict, hparam_domain_discrete, run_name)

    def close(self):
        for tag, entry in self.value_dict.items():
            if not isinstance(entry.value, list):
                super().add_scalar(tag, entry.value / entry.count, entry.global_step)
        super().close()
