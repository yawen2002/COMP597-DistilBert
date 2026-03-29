import src.config as config
import torch
import torch.utils.data
from transformers import AutoConfig, set_seed

DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
DISTILBERT_CATEGORY = "AutoModelForMaskedLM"
DISTILBERT_TRAIN_LENGTH = 512
DISTILBERT_EVAL_LENGTH = 512

data_load_name = "DistilBert"
generators = {}


def register_generator(fn):
    generators[fn.__name__.lstrip("gen_")] = fn
    return fn


class SyntheticData(torch.utils.data.Dataset):
    """Synthetic dataset adapted from MilaBench's Hugging Face benchmark."""

    def __init__(self, generators, n: int, repeat: int):
        self.n = n
        self.repeat = repeat
        self.generators = generators
        self.data = [self.gen() for _ in range(n)]

    def gen(self):
        return {name: gen() for name, gen in self.generators.items()}

    def __getitem__(self, i):
        return self.data[i % self.n]

    def __len__(self):
        return self.n * self.repeat


def vocabgen(model_config, train_length: int):
    def gen():
        return torch.randint(0, model_config.vocab_size, (train_length,))

    return gen


@register_generator
def gen_AutoModelForCausalLM(model_config, train_length: int):
    return {
        "input_ids": vocabgen(model_config, train_length),
        "labels": vocabgen(model_config, train_length),
    }


@register_generator
def gen_AutoModelForMaskedLM(model_config, train_length: int):
    return gen_AutoModelForCausalLM(model_config, train_length)


def load_data(conf: config.Config) -> torch.utils.data.Dataset:
    distilbert_conf = conf.model_configs.DistilBert
    set_seed(distilbert_conf.seed)
    model_config = AutoConfig.from_pretrained(DISTILBERT_MODEL_NAME)

    return SyntheticData(
        generators=generators[DISTILBERT_CATEGORY](
            model_config,
            DISTILBERT_TRAIN_LENGTH,
        ),
        n=conf.batch_size,
        repeat=distilbert_conf.repeat,
    )

