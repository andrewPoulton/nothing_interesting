import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import torch
    import dataset
    import fire
    import os
    import random
    import transformers
    import pickle
    from tqdm import tqdm
    from utils import generate_mask, load_config
    from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DataLoader
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import wandb
wandb.init(project="prescalenorm")


def init_dataset(data):
    return dataset.SemEvalDataset('data/flat_semeval5way_train.csv') 

def init_dataloader(dataset, batch_size=32, random=True):
    sampler = RandomSampler(dataset) if random else SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=dataset.collater, num_workers=4)
    return loader

def init_model(type_vocab_size):
    model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.config.token_type_embeddings = type_vocab_size 
    model.bert.embeddings.token_type_embeddings = torch.nn.Embedding(type_vocab_size, model.config.hidden_size)
    return model

def init_optimizer(model, config):
    return torch.optim.Adam(model.parameters(), lr = config.max_lr)


def train_epoch(loader, model, optimizer, lr_scheduler, config, cuda):
    loss_fn = torch.nn.CrossEntropyLoss()
    with tqdm(total=len(loader.batch_sampler)) as pbar:
        epoch_loss = 0.
        for i, batch in enumerate(loader):
            if cuda:
                batch.cuda()
            optimizer.zero_grad()
            mask = generate_mask(batch)
            logits = model(input_ids = batch.input, attention_mask = mask, token_type_ids = batch.token_type_ids)
            logits = logits[0]
            # import pdb; pdb.set_trace()
            loss = loss_fn(logits.view(-1, config.num_labels), batch.labels.view(-1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if batch.labels.size(0)>1:
                acc = accuracy_score(batch.labels.cpu(), logits.argmax(-1).squeeze().cpu().detach())
            else:
                acc = 0.
            # if torch._np.isnan(loss.item()):
            # import pdb; pdb.set_trace()
            epoch_loss += loss.item()
            if i % config.log_interval == 0:
                    wandb.log({"Test Accuracy": acc, "Test Loss": loss.item()})
            pbar.set_description(f'global_step: {lr_scheduler.last_epoch}| loss: {loss.item():.4f}| acc: {acc*100:.1f}%| epoch_av_loss: {epoch_loss/(i+1):.4f} |')
            pbar.update(1)
            if lr_scheduler.last_epoch > config.total_steps:
                break
        return epoch_loss/(i+1)
        
def main(data,config):
    config = load_config(config)
    dataset = init_dataset(data)
    dataset.train_percent = config.train_data_percent
    dataset.set_data_source(config.data_source)
    loader = init_dataloader(dataset, batch_size=config.batch_size)
    model = init_model(type_vocab_size=config.type_vocab_size)
    wandb.watch(model)
    model.train()
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    optimizer = init_optimizer(model, config)
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.total_steps)

    while lr_scheduler.last_epoch <= config.total_steps:
        av_epoch_loss =  train_epoch(loader, model, optimizer, lr_scheduler, config, cuda)
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'full_bert_model.pt'))
        print('av_epoch_loss', av_epoch_loss)




if __name__ == "__main__":
    fire.Fire(main)