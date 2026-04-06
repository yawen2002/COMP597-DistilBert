from codecarbon import OfflineEmissionsTracker
import codecarbon
import codecarbon.core.cpu
import logging
import os
import time
from typing import Optional

import pandas as pd
import src.config as config
import src.trainer.stats.base as base
from src.trainer.stats.codecarbon import SimpleFileOutput
from src.trainer.stats.training_runtime import write_training_runtime_csv
import torch

logger = logging.getLogger(__name__)

# Reuse the same CPU-tracking choice as the other CodeCarbon stats: disable
# psutil CPU tracking so CodeCarbon falls back to its non-psutil modes.
codecarbon.core.cpu.is_psutil_available = lambda: False

trainer_stats_name = "codecarbon_grouped"


class TaskOnlyFileOutput(SimpleFileOutput):
    """Write only CodeCarbon task rows to a single CSV file."""

    def out(self, _total, _delta):
        pass

    def task_out(self, data, _experiment_name: str):
        if len(data) == 0:
            return

        df = pd.DataFrame.from_records([dict(data_point.values) for data_point in data])
        df = df.dropna(axis=1, how="all")
        df.to_csv(self.save_file_path, index=False)


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning(
            "No device provided to codecarbon_grouped trainer stats. Using default PyTorch device"
        )
        device = torch.get_default_device()
    if "num_train_steps" not in kwargs:
        raise ValueError("codecarbon_grouped trainer stats requires num_train_steps.")

    grouped_conf = conf.trainer_stats_configs.codecarbon_grouped
    return CodeCarbonGroupedStats(
        device=device,
        run_num=grouped_conf.run_num,
        project_name=grouped_conf.project_name,
        output_dir=grouped_conf.output_dir,
        steps_per_group=grouped_conf.steps_per_group,
        num_train_steps=kwargs["num_train_steps"],
        model_name=conf.model,
        batch_size=conf.batch_size,
    )


class CodeCarbonGroupedStats(base.TrainerStats):
    """Provides grouped-step CodeCarbon measurements across the training run."""

    def __init__(
        self,
        device: torch.device,
        run_num: int,
        project_name: str,
        output_dir: str,
        steps_per_group: int,
        num_train_steps: int,
        model_name: str,
        batch_size: int,
    ) -> None:
        if steps_per_group <= 0:
            raise ValueError("steps_per_group must be positive.")
        if num_train_steps <= 0:
            raise ValueError("num_train_steps must be positive.")

        self.device = device
        self.run_num = run_num
        self.output_dir = output_dir
        self.steps_per_group = steps_per_group
        self.num_train_steps = num_train_steps
        self.model_name = model_name
        self.batch_size = batch_size
        self.iteration = 0
        self.current_group_name: Optional[str] = None
        self.train_start_ns = 0
        self.train_end_ns = 0

        os.makedirs(self.output_dir, exist_ok=True)

        run_number = f"run_{run_num}_"
        gpu_id = self.device.index
        grouped_output_file_name = f"{run_number}cc_grouped_rank_{gpu_id}.csv"
        self.grouped_output_path = os.path.join(self.output_dir, grouped_output_file_name)
        self.grouped_tracker = OfflineEmissionsTracker(
            project_name=project_name,
            experiment_name="groups",
            country_iso_code="CAN",
            region="quebec",
            save_to_file=False,
            output_handlers=[
                TaskOnlyFileOutput(
                    output_file_name=grouped_output_file_name,
                    output_dir=output_dir,
                )
            ],
            allow_multiple_runs=True,
            api_call_interval=-1,
            gpu_ids=[gpu_id],
            log_level="warning",
        )

        # Initialise the task-mode tracker before tasks are started.
        self.grouped_tracker.start()

    def _make_group_name(self, start_step: int) -> str:
        end_step = min(start_step + self.steps_per_group - 1, self.num_train_steps)
        return f"Steps #{start_step}-{end_step}"

    def start_train(self) -> None:
        self.iteration = 0
        self.current_group_name = None
        torch.cuda.synchronize(self.device)
        self.train_start_ns = time.perf_counter_ns()

    def stop_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self.train_end_ns = time.perf_counter_ns()
        if self.current_group_name is not None:
            self.grouped_tracker.stop_task(task_name=self.current_group_name)
            self.current_group_name = None
        self.grouped_tracker.stop()

    def start_step(self) -> None:
        self.iteration += 1
        if self.current_group_name is None:
            self.current_group_name = self._make_group_name(self.iteration)
            torch.cuda.synchronize(self.device)
            self.grouped_tracker.start_task(task_name=self.current_group_name)

    def stop_step(self) -> None:
        should_close_group = (
            self.current_group_name is not None
            and (
                self.iteration % self.steps_per_group == 0
                or self.iteration == self.num_train_steps
            )
        )
        if should_close_group:
            torch.cuda.synchronize(self.device)
            self.grouped_tracker.stop_task(task_name=self.current_group_name)
            self.current_group_name = None

    def start_forward(self) -> None:
        pass

    def stop_forward(self) -> None:
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        pass

    def start_backward(self) -> None:
        pass

    def stop_backward(self) -> None:
        pass

    def start_optimizer_step(self) -> None:
        pass

    def stop_optimizer_step(self) -> None:
        pass

    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        runtime_path = write_training_runtime_csv(
            output_dir=self.output_dir,
            run_num=self.run_num,
            model_name=self.model_name,
            batch_size=self.batch_size,
            runtime_ns=self.train_end_ns - self.train_start_ns,
        )
        logger.info("CODECARBON GROUPED LOGGING: Groups saved to %s", self.grouped_output_path)
        logger.info("TRAINING RUNTIME LOGGING: Summary saved to %s", runtime_path)
