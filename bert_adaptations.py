import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import torch
    import dataset
    import fire
    import os
    import random
    import transformers
    from models import VariableNormTransformerLayer, AdaNorm, ScaleNorm

def tie_query_key(model):
    for layer in model.bert.encoder.layer:
        layer.attention.self.query.weight = layer.attention.self.key.weight
        layer.attention.self.query.bias = layer.attention.self.key.bias
    return model

def bert_to_variable_layer(model, variable_layer_config):
    new_layers = [VariableNormTransformerLayer(variable_layer_config) for _ in model.bert.encoder.layer]
    [new_layer.load_from_bert(old_layer) for new_layer, old_layer in zip(new_layers, model.bert.encoder.layer)]
    for i in range(len(model.bert.encoder.layer)):
        model.bert.encoder.layer[i] = new_layers[i]
    return model
    
def replace_layer_norm(module, config):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.LayerNorm:
            if config.norm_type == 'adanorm':
                adanorm_scale = 0.3
                eps = 1e-10
                replacement_norm = AdaNorm(adanorm_scale, eps)
            elif config.norm_type == 'scale':
                replacement_norm = ScaleNorm(config.hidden_size ** 0.5)
            else:
                return None #Noop if different norm_type
            setattr(module, attr_str, replacement_norm)
    for n, ch in module.named_children():
        replace_layer_norm(ch, config)
    # don't return anythin here as recursed
    