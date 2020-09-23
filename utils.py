import json
import torch
from types import SimpleNamespace

def load_config(cfg):
    config = json.load(open(cfg, 'r'))
    return SimpleNamespace(**config)

def generate_mask(batch):
    return torch.where(batch.input.eq(0), batch.input, torch.ones_like(batch.input))