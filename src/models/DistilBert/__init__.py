# === import necessary modules ===
import src.models.DistilBert.model
import src.config as config # Configurations
import src.trainer as trainer # Trainer base class

# === import necessary external modules ===
from typing import Any, Dict, Optional, Tuple
import torch.utils.data as data

model_name = "DistilBert"

def init_model(conf : config.Config, dataset : data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    return model_init(conf, dataset)
