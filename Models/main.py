from log import get_logger

import os
import torch
import numpy as np

import argparse
from packaging import version

from Model.training import Training
from Model.model import Propaganda_Detection
from dataPreparation import Dataset_Preparation

import transformers
from transformers import AutoTokenizer

pytorch_version = version.parse(transformers.__version__)
assert pytorch_version >= version.parse('3.0.0'), \
    'We now only support transformers version >=3.0.0, but your version is {}'.format(pytorch_version)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device == torch.device('cuda')

if __name__ == '__main__':

    # https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER/blob/master/src/run_transformer_ner.py
    parser = argparse.ArgumentParser()

    # ADD ARGUEMENTS
    parser.add_argument("--model_run", default='Bert_Softmax', type=str,
                        help="valid values: Bert_Softmax")
    parser.add_argument("--model_type", default='bert-base-cased', type=str,
                        help="valid values: bert-base-cased")
    parser.add_argument("--tokenizer_type", default='bert-base-cased', type=str,
                        help="valid values: bert-base-cased")
    parser.add_argument("--training", default=0, type=int,
                        help="valid values: 1 for Training, 0 for Validation")
    parser.add_argument("--seed", default=42, type=int,
                        help='random seed')
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="maximum number of tokens allowed in each sentence")
    parser.add_argument("--validation_batch_size", default=16, type=int,
                        help="The batch size for evaluation.")
    parser.add_argument("--training_batch_size", default=16, type=int,
                        help="The batch size for training")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for optimizer.")
    parser.add_argument("--num_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight Decay for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--optimizer", default='AdamW', type=str,
                        help="valid values: AdamW, SGD")
    parser.add_argument("--scheduler", default='LinearWarmup', type=str,
                        help="valid values: LinearWarmup, LRonPlateau")
    parser.add_argument("--full_finetuning", default=1, type=int,
                        help="Update weights for all layers or finetune last classification layer")
    parser.add_argument("--debugging", default=1, type=int,
                        help="Debugging Mode")

    global_args = parser.parse_args()

    LogFileExist = os.path.exists(os.getcwd() + '/Log_Files')
    ModelFileExist = os.path.exists(os.getcwd() + '/Model_Files')
    if not LogFileExist:
        os.makedirs(os.getcwd() + '/Log_Files')
    if not ModelFileExist:
        os.makedirs(os.getcwd() + '/Model_Files')


    paths = {
            "Techniques":"./techniques.json",
            "Log_Folder":"./Log_Files/",
            "Model_Files":"./Model_Files/",
            "Model_Selection":"./Model_Selection/",
            "Training_Data": "./Data_Files/Splits/train_split.json",
            "Validation_Data": "./Data_Files/Splits/val_split.json",
            "Log_Folder":"./Log_Files/"
    }

    hyper_params = {
        "model_run": global_args.model_run,
        "model_type": global_args.model_type,
        "tokenizer_type": global_args.tokenizer_type,
        "training": bool(global_args.training),
        "max_seq_length": global_args.max_seq_length,
        "random_seed": global_args.seed,
        "training_batch_size": global_args.training_batch_size,
        "validation_batch_size": global_args.validation_batch_size,
        "learning_rate": global_args.learning_rate,
        "epsilon": global_args.adam_epsilon,
        "weight_decay": global_args.weight_decay,
        "epochs": global_args.num_epochs,
        "scheduler": global_args.scheduler,
        "optimizer": global_args.optimizer,
        "max_grad_norm": global_args.max_grad_norm,
        "full_finetuning": bool(global_args.full_finetuning),
        "debugging": bool(global_args.debugging),
        "log_file": None,
        "datetime": None
    }


    
    ################################################## SEEDS
    seed = hyper_params['random_seed']
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    from datetime import datetime
    current_datetime = datetime.now()
    date_time = current_datetime.strftime("%d-%m-%Y_%H:%M:%S")
    hyper_params['datetime'] = date_time
    

    ################################################## LOG FILE SET UP
    
    if not hyper_params["debugging"]:
        file_name = paths['Log_Folder'] + hyper_params['model_run'] + '-' + date_time #global_args.log_file
        hyper_params['log_file'] = file_name
        logger_meta = get_logger(name='META', file_name=file_name, type='meta')
        logger_progress = get_logger(name='PORGRESS', file_name=file_name, type='progress')
        logger_results = get_logger(name='RESULTS', file_name=file_name, type='results')
        for i, (k, v) in enumerate(hyper_params.items()):
            if i == (len(hyper_params) - 1):
                logger_meta.warning("{}: {}\n".format(k, v))
            else:
                logger_meta.warning("{}: {}".format(k, v))
    else:
        logger_meta = None
        logger_progress = None
        logger_results = None


    ################################################## Models 

    # 1. Translation with BERT trained on Memes
    # 2. mBERT                                  'bert-base-multilingual-cased'
    # 3. RUBERT
    # 4. XLM RoBerta                            'xlm-roberta-base'
    # 5. XLM RoBerta Roman Urdu fine-tuned      'Aimlab/xlm-roberta-roman-urdu-finetuned'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device == torch.device('cuda')

    techniques = Dataset_Preparation.read_techniques(paths['Techniques'])









    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< XLM RoBerta Roman Urdu >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if hyper_params['model_run'] == 'XLM_RoBerta_Roman_Urdu':
        checkpoint_model = hyper_params['model_type'] = 'Aimlab/xlm-roberta-roman-urdu-finetuned'
        checkpoint_tokenizer = hyper_params['tokenizer_type'] = 'Aimlab/xlm-roberta-roman-urdu-finetuned'
        ##################################################  MODEL + TOKENIZER
        if hyper_params['training']:

            tokenizer = AutoTokenizer.from_pretrained(checkpoint_tokenizer, do_lower_case = False)
            model = Propaganda_Detection(checkpoint_model=checkpoint_model, num_tags=len(techniques))
            model = model.to(device)
            print('##################################################')
            if not hyper_params["debugging"]:
                logger_progress.critical('Model + Tokenizer Initialized')

            ##################################################  DATA PROCESSING
            dataPrep = Dataset_Preparation(paths, tokenizer, hyper_params)
            train_dataloader, valid_dataloader = dataPrep.run()
            if not hyper_params["debugging"]:
                logger_progress.critical('Tokenizing sentences and encoding labels')
                logger_progress.critical('Data Loaders Created')


            ##################################################  TRAINING
            if not hyper_params["debugging"]:
                logger_progress.critical('Training Started')
            train = Training(paths, model, tokenizer, hyper_params, train_dataloader, valid_dataloader, techniques, logger_results)
            train.run()
            if not hyper_params["debugging"]:
                logger_progress.critical('Training Finished')
                logger_progress.critical('Model Saved')
        else:
            pass
            ################################################## INFERENCE
            # print('##################################################')
            # if not hyper_params["debugging"]:
            #     logger_progress.critical('Starting Inference')
            # inference = Inferencer(paths, checkpoint_tokenizer, checkpoint_model, hyper_params, techniques)
            # macro_f1, micro_f1 = inference.run()
            # if not hyper_params["debugging"]:
            #     logger_results.info('Macro F1-Score | Micro F1-Score :  {} | {}'.format(macro_f1, micro_f1))
            #     logger_progress.critical('Inference Ended')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<                 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


















    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Multilingual_BERT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if hyper_params['model_run'] == 'Multilingual_BERT':
        checkpoint_model = hyper_params['model_type'] = 'bert-base-multilingual-cased'
        checkpoint_tokenizer = hyper_params['tokenizer_type'] = 'bert-base-multilingual-cased'
        ##################################################  MODEL + TOKENIZER
        if hyper_params['training']:

            tokenizer = AutoTokenizer.from_pretrained(checkpoint_tokenizer, do_lower_case = False)
            model = Propaganda_Detection(checkpoint_model=checkpoint_model, num_tags=len(techniques))
            model = model.to(device)
            print('##################################################')
            if not hyper_params["debugging"]:
                logger_progress.critical('Model + Tokenizer Initialized')

            ##################################################  DATA PROCESSING
            dataPrep = Dataset_Preparation(paths, tokenizer, hyper_params)
            train_dataloader, valid_dataloader = dataPrep.run()
            if not hyper_params["debugging"]:
                logger_progress.critical('Tokenizing sentences and encoding labels')
                logger_progress.critical('Data Loaders Created')


            ##################################################  TRAINING
            if not hyper_params["debugging"]:
                logger_progress.critical('Training Started')
            train = Training(paths, model, tokenizer, hyper_params, train_dataloader, valid_dataloader, techniques, logger_results)
            train.run()
            if not hyper_params["debugging"]:
                logger_progress.critical('Training Finished')
                logger_progress.critical('Model Saved')
        else:
            pass
            ################################################## INFERENCE
            # print('##################################################')
            # if not hyper_params["debugging"]:
            #     logger_progress.critical('Starting Inference')
            # inference = Inferencer(paths, checkpoint_tokenizer, checkpoint_model, hyper_params, techniques)
            # macro_f1, micro_f1 = inference.run()
            # if not hyper_params["debugging"]:
            #     logger_results.info('Macro F1-Score | Micro F1-Score :  {} | {}'.format(macro_f1, micro_f1))
            #     logger_progress.critical('Inference Ended')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<                 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>














    #----------------------------------------------------------------------------------
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('Torch Version : ', torch.__version__)
    # if device == torch.device('cuda'): print('CUDA Version  : ', torch.version.cuda)
    # print('There are %d GPU(s) available.' % torch.cuda.device_count())
    # print('We will use the GPU:', torch.cuda.get_device_name(0))


    # BERT                              bert-base-cased
    # Multilingual_BERT                 bert-base-multilingual-cased
    # RuBERT
    # XLM_RoBerta                       xlm-roberta-base
    # XLM_RoBerta_Roman_Urdu            Aimlab/xlm-roberta-roman-urdu-finetuned
    
    # nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9

    script = """
    python main.py \
        --model_run XLM_RoBerta_Roman_Urdu \
        --training 1 \
        --model_type default \
        --tokenizer_type default \
        --max_seq_length 256 \
        --training_batch_size 12 \
        --validation_batch_size 12 \
        --learning_rate 5e-5 \
        --num_epochs 10 \
        --seed 42 \
        --adam_epsilon 1e-8 \
        --max_grad_norm 1.0 \
        --optimizer AdamW \
        --scheduler LinearWarmup \
        --full_finetuning 1 \
        --debugging 1
    """

