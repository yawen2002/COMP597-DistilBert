from typing import Dict, Optional, Tuple
import src.config as config
import src.trainer as trainer
import src.trainer.stats as trainer_stats
import torch.optim as optim
import torch.utils.data as data
import transformers

DISTILBERT_MODEL_NAME = "distilbert-base-uncased"

def init_distilbert_model(conf: config.Config) -> transformers.PreTrainedModel:
    transformers.set_seed(conf.model_configs.DistilBert.seed)
    model_config = transformers.AutoConfig.from_pretrained(DISTILBERT_MODEL_NAME)
    return transformers.AutoModelForMaskedLM.from_config(model_config)


def init_distilbert_optim(conf: config.Config, model: transformers.PreTrainedModel) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=conf.learning_rate)


def simple_trainer(
    conf: config.Config,
    model: transformers.PreTrainedModel,
    dataset: data.Dataset,
) -> Tuple[trainer.Trainer, Optional[Dict]]:
    loader = data.DataLoader(dataset, batch_size=conf.batch_size)
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

    if conf.trainer == "simple":
        return simple_trainer(conf, model, dataset)

    raise Exception(f"Unknown trainer type {conf.trainer}")