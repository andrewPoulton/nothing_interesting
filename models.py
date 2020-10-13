import torch.nn as nn
import torch
import transformers
import math
from transformers.modeling_bert import BertEmbeddings, BertSelfAttention, BertIntermediate, BertLayerNorm
from transformers.modeling_distilbert import create_sinusoidal_embeddings, DistilBertForSequenceClassification
# from attention_masks import generate_mask
from types import SimpleNamespace



class ScaleNorm(nn.Module):
    """ScaleNorm"""
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm

class AdaNorm(nn.Module):
    def __init__(self, adanorm_scale, eps):
        super(AdaNorm, self).__init__()
        self.adanorm_scale = adanorm_scale
        self.eps = eps

    def forward(self, inputs):
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True)
        inputs = inputs - mean
        mean = inputs.mean(-1, keepdim=True)
        graNorm = (1 / 10 * (inputs - mean) / (std + self.eps)).detach()
        input_norm = (inputs - inputs * graNorm) / (std + self.eps)
        return input_norm*self.adanorm_scale

class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class VariableNormEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(VariableNormEmbeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
       
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
        else:
            token_type_embeddings = torch.zeros_like(inputs_embeds)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class VariableNormTransformerLayer(nn.Module):
    def __init__(self, config):
        super(VariableNormTransformerLayer, self).__init__()
        self.config = config

        if self.config.norm_type == 'layer':
            self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        elif self.config.norm_type == 'adanorm':
            self.attention_norm = AdaNorm(0.3, config.layer_norm_eps)
        elif self.config.norm_type == 'scalenorm':
            self.attention_norm = ScaleNorm(config.hidden_size ** 0.5)

        
        self.self_attention = BertSelfAttention(config)
        self.self_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.config.norm_type == 'layer':
            self.ff_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        elif self.config.norm_type == 'adanorm':
            self.ff_norm = AdaNorm(0.3, config.layer_norm_eps)
        elif self.config.norm_type == 'scalenorm':
            self.ff_norm = ScaleNorm(config.hidden_size ** 0.5)

        self.ff1 = BertIntermediate(config)
        self.ff2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ff_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask = None, *args,  **kwargs):
        residual = hidden_states
        if self.config.prenorm:
            hidden_states = self.attention_norm(hidden_states)
        # Self-attention sublayers
        if not attention_mask is None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:,None,None,:]
        hidden_states, attentions = self.self_attention(hidden_states, attention_mask=attention_mask, output_attentions=self.config.output_attentions)
        hidden_states = self.self_out(hidden_states)
        hidden_states = self.self_dropout(hidden_states) + residual
        if not self.config.prenorm:
            hidden_states = self.attention_norm(hidden_states)

        residual = hidden_states
        if self.config.prenorm:
            hidden_states = self.ff_norm(hidden_states)
        # FF sublayer
        hidden_states = self.ff1(hidden_states)
        hidden_state = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.ff2(hidden_states)
        hidden_states = self.ff_dropout(hidden_states) + residual
        if not self.config.prenorm:
            hidden_states = self.ff_norm(hidden_states)
        
        return hidden_states, attentions

    def load_from_bert(self, bert_layer):
        self.self_attention.load_state_dict(bert_layer.attention.self.state_dict())
        self.self_out.load_state_dict(bert_layer.attention.output.dense.state_dict())
        self.ff1.load_state_dict(bert_layer.intermediate.state_dict())
        self.ff2.load_state_dict(bert_layer.output.dense.state_dict())
        if self.config.norm_type == "layer":
            self.attention_norm.load_state_dict(bert_layer.attention.output.LayerNorm.state_dict())
            self.ff_norm.load_state_dict(bert_layer.output.LayerNorm.state_dict())

class VariableNormTransformer(nn.Module):
    def __init__(self, config):
        super(VariableNormTransformer, self).__init__()
        self.config = config
        self.embeddings = VariableNormEmbeddings(config)
        self.transformer = nn.ModuleList([VariableNormTransformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler = Pooler(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self._init_weights)
  
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        

    def forward(self, batch, attention_mask, **kwargs):
        hidden_states = self.embeddings(batch.input, 
                                        batch.token_type_ids 
                                        )
        output_attentions = tuple()
        # additive mask!!
        attention_mask = (1.0 - attention_mask) * -10000.0
        for i, layer in enumerate(self.transformer):
            hidden_states, attentions = layer(hidden_states, attention_mask)
            output_attentions = output_attentions + (attentions,)
        hidden_states = self.pooler(hidden_states)
        logits = self.classifier(hidden_states)
        return logits, output_attentions

    def inspection_forward(self, word_embeddings, token_type_embeddings, position_embeddings, attention_mask):
        if self.embeddings.fix_norm:
            word_embeddings = nn.functional.normalize(word_embeddings, dim = -1)
            embeddings = word_embeddings + position_embeddings + token_type_embeddings
        else:
            embeddings = self.embeddings.LayerNorm(embeddings)

        hidden_states = self.embeddings.dropout(embeddings)
        attention_mask = (1.0 - attention_mask) * -10000.0
        for i, layer in enumerate(self.transformer):
            hidden_states, _ = layer(hidden_states, attention_mask)
        hidden_states = self.pooler(hidden_states)
        logits = self.classifier(hidden_states)
        return logits

    def format_batch_for_inspection(self, answer_text, ds_row, ds, zero_baseline = True):
        # eg.encoded_text = torch.Tensor(ds.tokenizer.encode(text = eg.question, text_pair = text)).long()
        encoded_answer = torch.LongTensor(ds.tokenizer.encode(answer_text).ids[1:])
        old_text_length = ds_row.token_types.eq(3).sum().item()
        prefix_length = ds_row.token_types.size(0) - old_text_length
        ds_row.encoded_text = torch.cat((ds_row.encoded_text[:prefix_length], encoded_answer))
        length_diff = old_text_length - encoded_answer.size(0)
        if length_diff > 0:
            ds_row.token_types = ds_row.token_types[:-length_diff]
        elif length_diff < 0:
            
            ds_row.token_types = torch.cat((ds_row.token_types, torch.LongTensor([3]*(-length_diff))))
        # import pdb; pdb.set_trace()
        batch = ds.collater([ds_row])
        baseline = torch.cat((ds_row.encoded_text[:prefix_length], torch.zeros_like(encoded_answer)))
        
        baseline = self.embeddings.word_embeddings(baseline.unsqueeze(0))
        if zero_baseline:
            baseline = torch.zeros_like(baseline).float()
        inputs = self.embeddings.word_embeddings(batch.input)

        position_ids = torch.arange(batch.input.size(1), dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(batch.input.shape)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.embeddings.token_type_embeddings(batch.token_type_ids)

        tokens = [ds.tokenizer.id_to_token(idx) for idx in ds_row.encoded_text.numpy()]
        mask = generate_mask(batch, "default")
        return inputs, mask, (token_type_embeddings, position_embeddings), tokens, baseline, batch