from src.config.util.base_config import _Arg, _BaseConfig

config_name = "codecarbon_grouped"


class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_run_num = _Arg(
            type=int,
            help="The run number used for grouped CodeCarbon file tracking.",
            default=0,
        )
        self._arg_project_name = _Arg(
            type=str,
            help="The project name used by CodeCarbon for grouped tracking.",
            default="energy-efficiency",
        )
        self._arg_output_dir = _Arg(
            type=str,
            help="The output directory where grouped CodeCarbon files will be saved.",
            default=".",
        )
        self._arg_steps_per_group = _Arg(
            type=int,
            help="The number of full training steps in each grouped CodeCarbon window; choose it so one window is slightly above 500 ms.",
            default=1,
        )
