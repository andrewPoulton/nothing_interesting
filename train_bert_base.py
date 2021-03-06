import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import torch
    import dataset
    import fire
    import os
    import json
    import random
    import transformers
    import pickle
    from tqdm import tqdm
    from utils import generate_mask, load_config, init_model, configure_model
    from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DataLoader
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import bert_adaptations
    import wandb
    import gc




def init_dataset(data):
    return dataset.SemEvalDataset(data) 

def init_dataloader(dataset, batch_size=32, random=True):
    sampler = RandomSampler(dataset) if random else SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=dataset.collater, num_workers=4)
    return loader



def init_optimizer(model, config):
    return torch.optim.Adam(model.parameters(), lr = config.max_lr)

def grad_norm(model):
    return sum([p.grad.pow(2).sum() if p.grad is not None else torch.tensor(0.) for p in model.parameters()])**.5 

def metrics(predictions, y_true, metric_params):
    precision = precision_score(y_true, predictions, **metric_params)
    recall = recall_score(y_true, predictions, **metric_params)
    f1 = f1_score(y_true, predictions, **metric_params)
    accuracy = accuracy_score(y_true, predictions)
    return precision, recall, f1, accuracy

@torch.no_grad()          
def val_loop(model, loader, cuda):
    model.eval()
    # batches = list(loader)
    preds = [] 
    true_labels = [] 
    with tqdm(total= len(loader.batch_sampler)) as pbar:
        for i,batch in enumerate(loader):
            if cuda:
                batch.cuda()
            mask = generate_mask(batch)
            logits = model(input_ids = batch.input, attention_mask = mask, token_type_ids = batch.token_type_ids)
            logits = logits[0]
            preds.append(logits.argmax(-1).squeeze().cpu())
            true_labels.append(batch.labels.cpu())
            pbar.update(1)
    preds = torch.cat(preds)
    y_true = torch.cat(true_labels)
    model.train()
    metric_params = {'average':'weighted', 'labels':list(range(model.config.num_labels))}
    return metrics(preds, y_true, metric_params)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            lr_scheduler.step()

           
            if batch.labels.size(0)>1:
                acc = accuracy_score(batch.labels.cpu(), logits.cpu().detach().argmax(-1).squeeze())
            else:
                acc = 0.
            # if torch._np.isnan(loss.item()):
            # import pdb; pdb.set_trace()
            epoch_loss += loss.item()
            # if i % config.log_interval == 0:
            wandb.log({"Train Accuracy": acc, "Train Loss": loss.item(), "Gradient Norm": grad_norm(model).item(), "Learning Rate": optimizer.param_groups[0]['lr']})
            pbar.set_description(f'global_step: {lr_scheduler.last_epoch}| loss: {loss.item():.4f}| acc: {acc*100:.1f}%| epoch_av_loss: {epoch_loss/(i+1):.4f} |')
            pbar.update(1)
            if lr_scheduler.last_epoch > config.total_steps:
                break
        #  move stuff off GPU
        batch.cpu()
        logits = logits.cpu().detach().argmax(-1).squeeze()
        return epoch_loss/(i+1)


        
def main(data, val_data, config):
    
    wandb.init(project="prescalenorm")
    config = load_config(config)
    dataset = init_dataset(data)
    dataset.train_percent = config.train_data_percent
    dataset.set_data_source(config.data_source)
    loader = init_dataloader(dataset, batch_size=config.batch_size)
    model = init_model(type_vocab_size=config.type_vocab_size)

    print(f'from_scratch: {config.from_scratch}', f'prenorm: {config.prenorm}', f'tie_qk: {config.tie_query_key}', f'norm_type: {config.norm_type}')

    model = configure_model(model, config)


    val_dataset = init_dataset(val_data)
    val_dataset.train_percent = config.val_data_percent
    val_dataset.to_val_mode('scientsbank', 'answer')
    val_loader = init_dataloader(val_dataset, batch_size=config.batch_size, random=False)

    wandb.watch(model)
    model.train()
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    optimizer = init_optimizer(model, config)
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.total_steps)
    # best_val_acc = 0.0
    # torch.save(config, os.path.join(wandb.run.dir, 'model.config'))
    json.dump(config.__dict__, open(os.path.join(wandb.run.dir, 'model_config.json'), 'w'))
    wandb.save('*.config')
    best_f1 = 0.0
    patience = 0
    try:
        while lr_scheduler.last_epoch <= config.total_steps:
            av_epoch_loss =  train_epoch(loader, model, optimizer, lr_scheduler, config, cuda)
            #tidy stuff up every epoch
            gc.collect()
            torch.cuda.empty_cache()

            p,r,f1,val_acc = val_loop(model, val_loader, cuda)
            log_line = f'precision: {p:.5f} | recall: {r:.5f} | f1: {f1:.5f} | accuracy: {val_acc:.5f}\n'
            print(log_line[:-1])
            print('av_epoch_loss', av_epoch_loss)
            if f1 > best_f1:
                print("saving to: ", os.path.join(wandb.run.dir, f'full_bert_model_best_acc.pt'))
                torch.save([model.state_dict(), config.__dict__], os.path.join(wandb.run.dir, f'full_bert_model_best_f1.pt'))
                wandb.save('*.pt')
                best_f1 = f1
                patience = max((0, patience-1))
            else:
                patience +=1
                if patience >= 3:
                    break
            if av_epoch_loss < .2:
                break
        torch.save([model.state_dict(), config.__dict__], os.path.join(wandb.run.dir, f'full_bert_model_{lr_scheduler.last_epoch}_steps.pt'))
        #Move stuff off the gpu
        model.cpu()
        #This is for sure a kinda dumb way of doing it, but the least mentally taxing right now
        optimizer = init_optimizer(model, config)
        gc.collect()
        torch.cuda.empty_cache()
        return model
        
    except KeyboardInterrupt:
        wandb.save('*.pt')
        #Move stuff off the gpu
        model.cpu()
        #This is for sure a kinda dumb way of doing it, but the least mentally taxing right now
        optimizer = init_optimizer(model, config)
        gc.collect()
        torch.cuda.empty_cache()
        return model
        
    
if __name__ == "__main__":
    fire.Fire(main)