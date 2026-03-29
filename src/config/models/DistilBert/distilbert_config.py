from src.config.util.base_config import _Arg, _BaseConfig


class ModelConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_seed = _Arg(
            type=int,
            help="Random seed for DistilBert synthetic data generation and model initialization.",
            default=1234,
        )
        self._arg_num_workers = _Arg(
            type=int,
            help="Number of DataLoader workers used for the DistilBert workload.",
            default=4,
        )
        self._arg_repeat = _Arg(
            type=int,
            help="How many times to repeat the cached synthetic DistilBert samples.",
            default=100000,
        )
