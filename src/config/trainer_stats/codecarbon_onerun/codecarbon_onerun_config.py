from src.config.util.base_config import _Arg, _BaseConfig

config_name = "codecarbon_onerun"


class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_run_num = _Arg(
            type=int,
            help="The run number used for end-to-end CodeCarbon file tracking.",
            default=0,
        )
        self._arg_project_name = _Arg(
            type=str,
            help="The project name used by CodeCarbon for one-run tracking.",
            default="energy-efficiency",
        )
        self._arg_output_dir = _Arg(
            type=str,
            help="The output directory where one-run CodeCarbon files will be saved.",
            default=".",
        )
