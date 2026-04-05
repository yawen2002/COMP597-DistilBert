from src.config.util.base_config import _Arg, _BaseConfig

config_name = "training_runtime"


class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_run_num = _Arg(
            type=int,
            help="Run number used to distinguish repeated training-runtime runs.",
            default=0,
        )
        self._arg_output_dir = _Arg(
            type=str,
            help="Directory where training-runtime CSV files will be written.",
            default=".",
        )
