import json
import torch
import bert_adaptations
import transformers
from types import SimpleNamespace

def load_config(cfg):
    config = json.load(open(cfg, 'r'))
    return SimpleNamespace(**config)

def generate_mask(batch):
    return torch.where(batch.input.eq(0), batch.input, torch.ones_like(batch.input))

def configure_model(model, config):
    if config.from_scratch:
        model.init_weights()

    if config.tie_query_key:
        model = bert_adaptations.tie_query_key(model)

    # set all layers to variable layers, inplace
    # this transfers all weights as well, skipping norm layers if incompatible
    # thus the resulting model is pre-norm if the config says so.
    # Note that key and query weights have to be tied before this
    model = bert_adaptations.bert_to_variable_layer(model, config)

    if (config.norm_type != 'layer'):
        bert_adaptations.replace_layer_norm(model, config)
    
    return model

def init_model(type_vocab_size):
    model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.config.token_type_embeddings = type_vocab_size 
    emb = torch.nn.Embedding(type_vocab_size, model.config.hidden_size)
    emb.weight.data.normal_(mean=0.0, std = model.config.initializer_range)  
    model.bert.embeddings.token_type_embeddings = emb
    return model

def load_model_from_state_dict(config, state_dict):
    model = init_model(config.type_vocab_size)
    model = configure_model(model, config)
    model.load_state_dict(state_dict)
    return model