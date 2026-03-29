from types import SimpleNamespace as NS
from typing import Dict, Optional, Tuple
import src.config as config
import src.trainer as trainer
import src.trainer.stats as trainer_stats
import torch.optim as optim
import torch.utils.data as data
import transformers


def init_distilbert_info() -> NS:
    return NS(
        category="AutoModelForMaskedLM",
        config=transformers.AutoConfig.from_pretrained("distilbert-base-uncased"),
        train_length=512,
        eval_length=512,
    )


def init_distilbert_model(conf: config.Config) -> transformers.PreTrainedModel:
    distilbert_conf = getattr(conf.model_configs, "DistilBert")
    seed = getattr(distilbert_conf, "seed")
    transformers.set_seed(seed)
    info = init_distilbert_info()
    return getattr(transformers, info.category).from_config(info.config)


def init_distilbert_optim(conf: config.Config, model: transformers.PreTrainedModel) -> optim.Optimizer:
    learning_rate = getattr(conf, "learning_rate")
    return optim.Adam(model.parameters(), lr=learning_rate)


def simple_trainer(
    conf: config.Config,
    model: transformers.PreTrainedModel,
    dataset: data.Dataset,
) -> Tuple[trainer.Trainer, Optional[Dict]]:
    distilbert_conf = getattr(conf.model_configs, "DistilBert")
    batch_size = getattr(conf, "batch_size")
    num_workers = getattr(distilbert_conf, "num_workers")
    loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    model = model.cuda()
    optimizer = init_distilbert_optim(conf, model)
    scheduler = transformers.get_scheduler("constant", optimizer=optimizer)

    return trainer.SimpleTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=model.device,
        stats=trainer_stats.init_from_conf(
            conf=conf,
            device=model.device,
            num_train_steps=len(loader),
        ),
    ), None


def model_init(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    model = init_distilbert_model(conf)
    trainer_name = getattr(conf, "trainer")

    if trainer_name == "simple":
        return simple_trainer(conf, model, dataset)

    raise Exception(f"Unknown trainer type {trainer_name}")