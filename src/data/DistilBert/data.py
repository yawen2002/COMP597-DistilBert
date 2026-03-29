from types import SimpleNamespace as NS
import src.config as config
import torch
import torch.utils.data
from transformers import AutoConfig, set_seed

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
        self.data = self.gen(generators, n)

    @staticmethod
    def gen(generators, n: int):
        return {
            name: torch.stack([gen() for _ in range(n)])
            for name, gen in generators.items()
        }

    def __getitem__(self, i):
        idx = i % self.n
        return {name: values[idx] for name, values in self.data.items()}

    def __len__(self):
        return self.n * self.repeat


def vocabgen(info):
    def gen():
        return torch.randint(0, info.config.vocab_size, (info.train_length,))

    return gen


@register_generator
def gen_AutoModelForCausalLM(info):
    return {
        "input_ids": vocabgen(info),
        "labels": vocabgen(info),
    }


@register_generator
def gen_AutoModelForMaskedLM(info):
    return gen_AutoModelForCausalLM(info)


def load_data(conf: config.Config) -> torch.utils.data.Dataset:
    distilbert_conf = getattr(conf.model_configs, "DistilBert")
    batch_size = getattr(conf, "batch_size")
    seed = getattr(distilbert_conf, "seed")
    repeat = getattr(distilbert_conf, "repeat")

    set_seed(seed)
    distilbert_info = NS(
        category="AutoModelForMaskedLM",
        config=AutoConfig.from_pretrained("distilbert-base-uncased"),
        train_length=512,
        eval_length=512,
    )

    return SyntheticData(
        generators=generators[distilbert_info.category](distilbert_info),
        n=batch_size,
        repeat=repeat,
    )

