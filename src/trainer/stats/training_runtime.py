import csv
import logging
import os
import time

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.noop as noop
import torch

logger = logging.getLogger(__name__)

trainer_stats_name = "training_runtime"


def training_runtime_csv_path(output_dir: str, run_num: int) -> str:
    return os.path.join(output_dir, f"run_{run_num}_training_runtime.csv")


def write_training_runtime_csv(
    output_dir: str,
    run_num: int,
    model_name: str,
    batch_size: int,
    runtime_ns: int,
) -> str:
    path = training_runtime_csv_path(output_dir, run_num)
    row = {
        "run_num": run_num,
        "model": model_name,
        "batch_size": batch_size,
        "runtime_ns": runtime_ns,
        "runtime_s": runtime_ns / 1_000_000_000,
    }

    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    return path


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning(
            "No device provided to training_runtime trainer stats. Using default PyTorch device"
        )
        device = torch.get_default_device()

    runtime_conf = conf.trainer_stats_configs.training_runtime
    return TrainingRuntimeStats(
        device=device,
        run_num=runtime_conf.run_num,
        output_dir=runtime_conf.output_dir,
        model_name=conf.model,
        batch_size=conf.batch_size,
    )


class TrainingRuntimeStats(noop.NOOPTrainerStats):
    """Record training-only wallclock time between start_train and stop_train."""

    def __init__(
        self,
        device: torch.device,
        run_num: int,
        output_dir: str,
        model_name: str,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.run_num = run_num
        self.output_dir = output_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.train_start_ns = 0
        self.train_end_ns = 0

        os.makedirs(self.output_dir, exist_ok=True)

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start_train(self) -> None:
        self._sync_device()
        self.train_start_ns = time.perf_counter_ns()

    def stop_train(self) -> None:
        self._sync_device()
        self.train_end_ns = time.perf_counter_ns()

    def log_stats(self) -> None:
        runtime_path = write_training_runtime_csv(
            output_dir=self.output_dir,
            run_num=self.run_num,
            model_name=self.model_name,
            batch_size=self.batch_size,
            runtime_ns=self.train_end_ns - self.train_start_ns,
        )
        logger.info("TRAINING RUNTIME LOGGING: Summary saved to %s", runtime_path)
