from codecarbon import OfflineEmissionsTracker
import codecarbon
import codecarbon.core.cpu
import logging
import os

import src.config as config
import src.trainer.stats.base as base
from src.trainer.stats.codecarbon import SimpleFileOutput
import torch

logger = logging.getLogger(__name__)

# artificially force psutil to fail, so that CodeCarbon uses constant mode for CPU measurements
codecarbon.core.cpu.is_psutil_available = lambda: False

trainer_stats_name = "codecarbon_onerun"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning(
            "No device provided to codecarbon_onerun trainer stats. Using default PyTorch device"
        )
        device = torch.get_default_device()

    one_run_conf = conf.trainer_stats_configs.codecarbon_onerun
    return CodeCarbonOneRunStats(
        device=device,
        run_num=one_run_conf.run_num,
        project_name=one_run_conf.project_name,
        output_dir=one_run_conf.output_dir,
    )


class CodeCarbonOneRunStats(base.TrainerStats):
    """Provides one end-to-end CodeCarbon measurement for the full training run."""

    def __init__(
        self,
        device: torch.device,
        run_num: int,
        project_name: str,
        output_dir: str,
    ) -> None:
        self.device = device
        self.run_num = run_num
        self.project_name = project_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        run_number = f"run_{run_num}_"
        gpu_id = self.device.index

        # Normal-mode tracker to track the entire training loop
        self.total_training_tracker = OfflineEmissionsTracker(
            project_name=project_name,
            country_iso_code="CAN",
            region="quebec",
            save_to_file=False,
            output_handlers=[
                SimpleFileOutput(
                    output_file_name=f"{run_number}cc_onerun_rank_{gpu_id}.csv",
                    output_dir=output_dir,
                )
            ],
            allow_multiple_runs=True,
            log_level="warning",
            gpu_ids=[gpu_id],
        )

    def start_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self.total_training_tracker.start()

    def stop_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self.total_training_tracker.stop()

    def start_step(self) -> None:
        pass

    def stop_step(self) -> None:
        pass

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
        pass
