import csv
import logging
import os
import statistics
import threading
import time
from typing import Any, Dict, Optional

import psutil
import pynvml
import torch

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.simple as simple
from src.trainer.stats.training_runtime import write_training_runtime_csv

logger = logging.getLogger(__name__)

trainer_stats_name = "profile"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to profile trainer stats. Using default PyTorch device")
        device = torch.get_default_device()

    profile_conf = conf.trainer_stats_configs.profile
    return ProfileTrainerStats(
        device=device,
        run_num=profile_conf.run_num,
        output_dir=profile_conf.output_dir,
        sample_interval_ms=profile_conf.sample_interval_ms,
        model_name=conf.model,
        batch_size=conf.batch_size,
    )


class ProfileTrainerStats(simple.SimpleTrainerStats):
    """Fine-grained timing and utilization measurements.

    This stats class reuses the starter code's synchronized phase timing pattern
    and adds coarse timeline sampling for CPU utilization, process RSS memory,
    GPU utilization, and GPU memory. Timeline samples are taken at a fixed
    wall-clock interval configured by ``sample_interval_ms`` so the system-level
    measurements stay time-based and low overhead.
    """

    def __init__(
        self,
        device: torch.device,
        run_num: int,
        output_dir: str,
        sample_interval_ms: int,
        model_name: str,
        batch_size: int,
    ) -> None:
        super().__init__(device=device)
        if device.type != "cuda":
            raise ValueError("ProfileTrainerStats requires a CUDA device.")
        if sample_interval_ms <= 0:
            raise ValueError("sample_interval_ms must be positive.")
        self.device = device
        self.run_num = run_num
        self.output_dir = output_dir
        self.sample_interval_ms = sample_interval_ms
        self.sample_interval_s = sample_interval_ms / 1000.0
        self.model_name = model_name
        self.batch_size = batch_size
        self.current_step = 0
        self.current_state = "idle"
        self.last_loss: Optional[float] = None
        self.step_rows: list[Dict[str, Any]] = []
        self.timeline_rows: list[Dict[str, Any]] = []
        self.train_start_ns = 0
        self.train_end_ns = 0
        self.last_step_end_ns = 0
        self.stop_event = threading.Event()
        self.timeline_thread: Optional[threading.Thread] = None
        self.process = psutil.Process()

        os.makedirs(self.output_dir, exist_ok=True)

        pynvml.nvmlInit()
        self.handle = self._get_nvml_handle()

    def _get_nvml_handle(self) -> Any:
        logical_gpu_index = 0 if self.device.index is None else self.device.index
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices is not None:
            visible_device_list = [
                entry.strip() for entry in visible_devices.split(",") if entry.strip()
            ]
            if logical_gpu_index < len(visible_device_list):
                mapped_gpu = visible_device_list[logical_gpu_index]
                try:
                    physical_gpu_index = int(mapped_gpu)
                    return pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_index)
                except ValueError:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByUUID(mapped_gpu.encode("utf-8"))
                        return handle
                    except (AttributeError, pynvml.NVMLError):
                        logger.warning(
                            "CUDA_VISIBLE_DEVICES entry '%s' could not be mapped through NVML; "
                            "falling back to logical GPU index %d.",
                            mapped_gpu,
                            logical_gpu_index,
                        )

        return pynvml.nvmlDeviceGetHandleByIndex(logical_gpu_index)

    def _timeline_path(self) -> str:
        return os.path.join(self.output_dir, f"run_{self.run_num}_profile_timeline.csv")

    def _steps_path(self) -> str:
        return os.path.join(self.output_dir, f"run_{self.run_num}_profile_steps.csv")

    def _summary_path(self) -> str:
        return os.path.join(self.output_dir, f"run_{self.run_num}_profile_summary.csv")

    def _write_csv(self, path: str, fieldnames: list[str], rows: list[Dict[str, Any]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _append_timeline_row(
        self,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        now_ns = time.perf_counter_ns() if timestamp_ns is None else timestamp_ns
        step = self.current_step
        state = self.current_state
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        process_cpu_percent = self.process.cpu_percent(interval=None)
        process_memory = self.process.memory_info()
        self.timeline_rows.append(
            {
                "run_num": self.run_num,
                "model": self.model_name,
                "batch_size": self.batch_size,
                "timestamp_ns": now_ns,
                "time_since_start_s": (now_ns - self.train_start_ns) / 1_000_000_000,
                "step": step,
                "state": state,
                "process_cpu_percent": process_cpu_percent,
                "process_memory_rss_bytes": process_memory.rss,
                "gpu_util_pct": gpu_util.gpu,
                "gpu_mem_used_mb": gpu_memory.used / (1024 * 1024),
            }
        )

    def _sample_timeline(self) -> None:
        wait_for_start_s = min(self.sample_interval_s, 0.001)
        while self.train_start_ns == 0:
            if self.stop_event.wait(wait_for_start_s):
                return
        while not self.stop_event.wait(self.sample_interval_s):
            self._append_timeline_row()

    def _summary_stats(self, values: list[int]) -> tuple[float, float]:
        if len(values) == 0:
            return 0.0, 0.0
        if len(values) == 1:
            return float(values[0]), 0.0
        return float(statistics.mean(values)), float(statistics.stdev(values))

    def start_train(self) -> None:
        self.current_state = "train"
        self.current_step = 0
        self.last_loss = None
        self.last_step_end_ns = 0
        self.train_start_ns = 0
        self.process.cpu_percent(interval=None)
        self.stop_event.clear()
        self.timeline_thread = threading.Thread(target=self._sample_timeline, daemon=True)
        self.timeline_thread.start()
        torch.cuda.synchronize(self.device)
        self.train_start_ns = time.perf_counter_ns()

    def stop_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self.train_end_ns = time.perf_counter_ns()
        self.current_state = "finished"
        self._append_timeline_row(timestamp_ns=self.train_end_ns)
        self.stop_event.set()
        if self.timeline_thread is not None:
            self.timeline_thread.join()

    def start_step(self) -> None:
        self.current_step += 1
        self.current_state = "step"
        super().start_step()

    def stop_step(self) -> None:
        super().stop_step()
        self.last_step_end_ns = time.perf_counter_ns()
        self.current_state = "train"

    def start_forward(self) -> None:
        self.current_state = "forward"
        super().start_forward()

    def stop_forward(self) -> None:
        super().stop_forward()
        self.current_state = "step"

    def start_backward(self) -> None:
        self.current_state = "backward"
        super().start_backward()

    def stop_backward(self) -> None:
        super().stop_backward()
        self.current_state = "step"

    def start_optimizer_step(self) -> None:
        self.current_state = "optimizer"
        super().start_optimizer_step()

    def stop_optimizer_step(self) -> None:
        super().stop_optimizer_step()
        self.current_state = "step"

    def log_loss(self, loss: torch.Tensor) -> None:
        self.last_loss = float(loss.detach().item())

    def log_step(self) -> None:
        step_end_ns = self.last_step_end_ns
        if step_end_ns == 0:
            step_end_ns = time.perf_counter_ns()
        step_time_ns = self.step_stats.get_last()
        self.step_rows.append(
            {
                "run_num": self.run_num,
                "model": self.model_name,
                "batch_size": self.batch_size,
                "step": self.current_step,
                "step_end_time_s": (step_end_ns - self.train_start_ns) / 1_000_000_000,
                "loss": self.last_loss,
                "step_time_ns": step_time_ns,
                "forward_time_ns": self.forward_stats.get_last(),
                "backward_time_ns": self.backward_stats.get_last(),
                "optimizer_time_ns": self.optimizer_step_stats.get_last(),
            }
        )

    def log_stats(self) -> None:
        runtime_ns = self.train_end_ns - self.train_start_ns
        timeline_rows = sorted(
            (row for row in self.timeline_rows if row["timestamp_ns"] <= self.train_end_ns),
            key=lambda row: row["timestamp_ns"],
        )
        forward_mean, forward_stdev = self._summary_stats(self.forward_stats.stat.history)
        backward_mean, backward_stdev = self._summary_stats(self.backward_stats.stat.history)
        optimizer_mean, optimizer_stdev = self._summary_stats(self.optimizer_step_stats.stat.history)

        summary_rows = [
            {
                "run_num": self.run_num,
                "model": self.model_name,
                "batch_size": self.batch_size,
                "sample_interval_ms": self.sample_interval_ms,
                "runtime_ns": runtime_ns,
                "runtime_s": runtime_ns / 1_000_000_000,
                "steps_completed": len(self.step_rows),
                "forward_time_mean_ns": forward_mean,
                "forward_time_stdev_ns": forward_stdev,
                "backward_time_mean_ns": backward_mean,
                "backward_time_stdev_ns": backward_stdev,
                "optimizer_time_mean_ns": optimizer_mean,
                "optimizer_time_stdev_ns": optimizer_stdev,
            }
        ]

        if timeline_rows:
            self._write_csv(
                self._timeline_path(),
                list(timeline_rows[0].keys()),
                timeline_rows,
            )
        if self.step_rows:
            self._write_csv(
                self._steps_path(),
                list(self.step_rows[0].keys()),
                self.step_rows,
            )
        self._write_csv(
            self._summary_path(),
            list(summary_rows[0].keys()),
            summary_rows,
        )
        runtime_path = write_training_runtime_csv(
            output_dir=self.output_dir,
            run_num=self.run_num,
            model_name=self.model_name,
            batch_size=self.batch_size,
            runtime_ns=runtime_ns,
        )

        logger.info("PROFILE LOGGING: Summary saved to %s", self._summary_path())
        logger.info("PROFILE LOGGING: Step timings saved to %s", self._steps_path())
        logger.info("PROFILE LOGGING: Timeline samples saved to %s", self._timeline_path())
        logger.info("TRAINING RUNTIME LOGGING: Summary saved to %s", runtime_path)

        pynvml.nvmlShutdown()
