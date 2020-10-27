import torch
import dataset as ds
import os
from types import SimpleNamespace
from tqdm import tqdm
from utils import load_model_from_state_dict
from torch.utils.data import SequentialSampler, BatchSampler, DataLoader
from captum.attr import IntegratedGradients

__CUDA__ = torch.cuda.is_available()

def load_model():
    state_dict, config = torch.load('full_bert_model_best_f1.pt', map_location=None if __CUDA__ else 'cpu' )
    config =  SimpleNamespace(**config)
    model = load_model_from_state_dict(config, state_dict)
    return model, config
    
def make_filename(config):
    norm = config.norm_type
    prenorm = config.prenorm
    from_scratch = config.from_scratch
    fname = f'{norm}{"_pn_" if prenorm else ""}{"_fromscratch_" if from_scratch else "_pretrained_"}'
    current_results = [f for f in os.listdir('results') if f.startswith(fname)]
    return f'results/{fname}_{len(current_results)}.result'



def get_val_dataloader(source = 'scientsbank', origin = 'answer'):
    dataset = ds.SemEvalDataset('data/flat_semeval5way_test.csv')
    dataset.to_val_mode(source, origin)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=1, drop_last=False)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=dataset.collater, num_workers=4)
    return loader

def inspection_forward(inputs_embeds, model, batch):
    return model(inputs_embeds=inputs_embeds,token_type_ids = batch.token_type_ids)[0]

def inspection_parameters(model, batch):
    baseline = torch.where(batch.token_type_ids.eq(3), torch.zeros_like(batch.input), batch.input)
    baseline = model.bert.embeddings.word_embeddings(baseline)
    inputs_embeds = model.bert.embeddings.word_embeddings(batch.input)
    answer_length = batch.token_type_ids.eq(3).sum().cpu().item()
    return inputs_embeds, baseline, answer_length

def token_significance(attentions):
    return sum([o.sum(-2).sum(1).squeeze() for o in attentions]).cpu()

def calculate_attributes(model, val_loader, debug = False):
    model.eval()
    attributes = {
        'batch_number': [],
        'batch_ids': [],
        'token_significance': [],
        'integrated_gradients': [],
    }

    ig = IntegratedGradients(inspection_forward)
    with tqdm(total= len(val_loader.batch_sampler)) as pbar:
        for i, batch in enumerate(val_loader):
            if __CUDA__: batch.cuda()
            inputs_embeds ,baseline, answer_length = inspection_parameters(model, batch)
            pbar.set_description('Calculating significance')
            with torch.no_grad(): 
                preds, attentions = model(inputs_embeds = inputs_embeds, token_type_ids = batch.token_type_ids, output_attentions = True)
                predicted_class = preds.cpu().squeeze().argmax().item()
                significance = token_significance(attentions)
            pbar.set_description('Calculating integrated gradients...')
            integrated_gradients = ig.attribute(inputs_embeds, 
                                    baselines=baseline, 
                                    target = predicted_class,
                                    additional_forward_args=(model, batch), 
                                    n_steps=200, 
                                    internal_batch_size=24)
            integrated_gradients = integrated_gradients[0].sum(-1).detach().cpu().numpy()
            attributes['batch_number'].append(i)
            attributes['batch_ids'].append(batch.input.cpu().numpy().tolist())
            attributes['token_significance'].append(significance)
            attributes['integrated_gradients'].append(integrated_gradients)
            pbar.update(1)
            if debug and (i>10):
                break
    return attributes


def main():
    model, config = load_model()
    fname = make_filename(config)
    if __CUDA__: model.cuda()
    val_loader = get_val_dataloader()
    attributes = calculate_attributes(model, val_loader)
    torch.save([attributes, config.__dict__], fname)

if __name__ == "__main__":
    main()