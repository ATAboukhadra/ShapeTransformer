from __future__ import annotations

import os

import torch.nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from multigpu_helpers.avg_summary_writer import AvgSummaryWriter
from multigpu_helpers.logger import Logger


class DistributedHelper:
    __slots__ = "is_distributed", "world_size", "rank", "local_rank", "master_addr", "master_port", "is_master",\
                "conf", "logger", "train_summary_writer", "val_summary_writer"

    def __init__(self, backend: str = "nccl"):
        if dist.is_available():
            dist.init_process_group(backend)
        # mp.set_start_method("forkserver")

        self.is_distributed = dist.is_initialized()
        self.world_size = int(os.environ["WORLD_SIZE"]) if self.is_distributed else 1
        self.rank = int(os.environ["RANK"]) if self.is_distributed else 0
        self.local_rank = int(os.environ["LOCAL_RANK"]) if self.is_distributed else 0

        self.master_addr = str(os.environ["MASTER_ADDR"]) if self.is_distributed else "localhost"
        self.master_port = int(os.environ["MASTER_PORT"]) if self.is_distributed else "1234"
        self.is_master = self.rank == 0

        self.logger: Logger | None = None
        self.train_summary_writer: AvgSummaryWriter | None = None
        self.val_summary_writer: SummaryWriter | None = None

    def wrap_model_for_ddp(self, model: torch.nn.Module):
        model = model.to(self.local_rank)
        if dist.is_initialized():
            model = DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)
        return model

    def sync_distributed_values(self, values: dict[str, torch.Tensor]):
        if self.is_distributed:
            for key in values.keys():
                dist.reduce(values[key], 0, dist.ReduceOp.AVG)

    def write_log(self, split: str, log_dict: dict, epoch: int):
        if self.logger is not None:
            message = f"epoch {epoch+1}, {split}:"
            for k, v in log_dict.items():
                message += f" {k} {v:8f} |"
            self.logger.write(message[:len(message)-1] + "\n")

    def flush_log(self):
        if self.logger is not None:
            self.logger.flush()

    def write_tb_train_log(self, tb_log: dict[str, float], global_step: int):
        if self.train_summary_writer is not None:
            for k, v in tb_log.items():
                self.train_summary_writer.add_scalar(k, v, global_step)

    def write_tb_val_log(self, tb_log: dict[str, float], global_step: int):
        if self.val_summary_writer is not None:
            for k, v in tb_log.items():
                self.val_summary_writer.add_scalar(k, v, global_step)
