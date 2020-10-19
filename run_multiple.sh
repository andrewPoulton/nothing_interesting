#! /bin/bash

python train_bert_base.py --data 'data/flat_semeval5way_train.csv' --val_data 'data/flat_semeval5way_test.csv' --config 'configs/finetune_config.json'
python train_bert_base.py --data 'data/flat_semeval5way_train.csv' --val_data 'data/flat_semeval5way_test.csv' --config 'configs/from_scratch_config.json'
python train_bert_base.py --data 'data/flat_semeval5way_train.csv' --val_data 'data/flat_semeval5way_test.csv' --config 'configs/prescalenorm_config.json'