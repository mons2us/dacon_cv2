import os
import csv
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import KFold
import random
from time import time
import IPython
import copy

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#from torchvision.models import resnet50

from warmup_scheduler import GradualWarmupScheduler

from src.train import train_model
from utils.imageprocess import image_transformer, image_processor
from utils.EarlyStopping import EarlyStopping
from utils.dataloader import CustomDataLoader
from utils.radams import RAdam
from utils.call_model import CallModel

#from utils.pretrained_model import CustomModel, CallPretrainedModel

from tqdm import tqdm
import logging


# train/val splitting method
def split_index(total_index, val_ratio):
    tot_len = len(total_index)
    train_len = int(tot_len*(1-val_ratio))

    train_sampled = random.sample(total_index, train_len)
    val_sampled = [i for i in total_index if i not in train_sampled]
    logger.info(f"Trainset length: {len(train_sampled)}, Valset length: {len(val_sampled)}")
    
    return train_sampled, val_sampled


def split_kfold(k, train_len=50000):    
    kfold = KFold(n_splits=k, shuffle=True)
    splitted = kfold.split(range(train_len))
    return splitted


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def load_dataset(mode='train', **kwargs):
    
    img_dir = kwargs['img_dir']
    label_dir = kwargs['label_dir']

    if mode=='train':
        
        train_index = kwargs['train_index']
        val_index = kwargs['val_index']
        batch_size = kwargs['batch_size']
        
        train_set = CustomDataLoader(
            img_dir=img_dir,
            label_dir=label_dir,
            train=True,
            row_index=train_index,
            device=global_device)

        val_set = CustomDataLoader(
            img_dir=img_dir,
            label_dir=label_dir,
            train=False,
            row_index=val_index,
            device=global_device)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)

        return train_loader, val_loader

    else:
        test_index = kwargs['test_index']
        tta = kwargs['tta']
        
        if not tta:
            tmp_test_set = CustomDataLoader(
                img_dir=img_dir,
                label_dir=label_dir,
                train=False,
                row_index=test_index,
                device=global_device)
            tmp_test_loader = torch.utils.data.DataLoader(tmp_test_set, batch_size=8, shuffle=False)
            test_loader = [tmp_test_loader]
        
        else:
            angles = kwargs['angles']
            
            test_loader = []
            for angle in angles:
                tmp_test_set = CustomDataLoader(
                    img_dir=img_dir,
                    label_dir=label_dir,
                    row_index=test_index,
                    device=global_device,
                    tta=True,
                    angle=angle)
                
                tmp_test_loader = torch.utils.data.DataLoader(tmp_test_set, batch_size=8, shuffle=False)
                test_loader.append(tmp_test_loader)

        #test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

        return test_loader


def load_trained_weight(model_input=None, model_index=0, model_type='early', fold_k=1, trained_weight_path='./ckpt'):

    assert model_index > 0

    model_name = f'early_stopped_fold{fold_k}.pth' if model_type == 'early' else f'model_ckpt_fold{fold_k}_{model_type}.pth'
    ckpt_path = os.path.join(trained_weight_path, f'model_{model_index}', model_name)

    trained_model = model_input
    #IPython.embed(); exit(1)
    trained_model.load_state_dict(torch.load(ckpt_path))
    trained_model.eval()

    return trained_model
    

def make_inference(args, model, test_loader):
    
    total_set = []
    for idx, single_test_loader in enumerate(test_loader):
        logger.info(f"Inference on test_loader ({idx+1}/{len(test_loader)})")
        fin_labels = []
        for _, (test_X, test_Y) in enumerate(single_test_loader):
            
            pred = model(test_X)
            #IPython.embed(); exit(1)
            pred_label = ((pred > args.threshold)*1).detach().to('cpu').numpy()
            fin_labels.append(pred_label)
        
        logger.info("Done.")
        fin_labels = np.concatenate(fin_labels)
        total_set.append(fin_labels)
    
    #IPython.embed(); exit(1)
    fin_total_set = np.where(np.mean(total_set, axis=0) > 0.5, 1, 0)
    
    return fin_total_set
 
    
def aggregate_submit(args, predictions):
    
    submission_file = pd.read_csv(os.path.join(args.base_dir, 'dataset/sample_submission.csv'))

    # Aggregation(voting)
    agg = np.where(np.mean(predictions, axis=0) >= 0.5, 1, 0)
    
    # Predictions / Submit
    submission_file.iloc[:, 1:] = agg

    save_path = os.path.join(args.base_dir, f'submit/submission_model_{args.model_index}.csv')

    submission_file.to_csv(save_path, index=False)
    logger.info(f'Result file save at: {save_path}')



if __name__ == "__main__":
    
    # ARGUMENTS PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_index", type=int, default=0, help='My model index. Integer type, and should be greater than 0')
    parser.add_argument("--base_dir", type=str, default="/home/sks/COMPETITION/DACON/computer_vision2", help='Base PATH of your work')
    parser.add_argument("--mode", type=str, default="train", help='[train | test]')
    parser.add_argument("--data_type", type=str, default="denoised", help='[original | denoised]: default=denoised')
    parser.add_argument("--augmix", dest='augmix', action='store_true', help='Whether to use augmix. Specify this argument to use augmix')
    parser.add_argument("--ckpt_path", type=str, default="/home/sks/COMPETITION/DACON/computer_vision2/ckpt", help='PATH to weights of ckpts.')
    parser.add_argument("--base_model", type=str, default="resnet50", help="[plain_resnet50, custom_resnet50, plain_efficientnetb4]")
    parser.add_argument("--pretrained", dest='pretrained', action='store_true', help='Default is false, so specify this argument to use pretrained model')
    parser.add_argument("--pretrained_weights_dir", type=str, default="/home/sks/COMPETITION/DACON/computer_vision2/pretrained_model", help='PATH to weights of pretrained model')
    parser.add_argument("--cuda", dest='cuda', action='store_false', help='Whether to use CUDA: defuault is True, so specify this argument not to use CUDA')
    parser.add_argument("--device_index", type=int, default=0, help='Cuda device to use. Used for multiple gpu environment')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size for train-loader for training phase')
    parser.add_argument("--val_ratio", type=float, default=0.1, help='Ratio for validation set: default=0.1')
    parser.add_argument("--epochs", type=int, default=100, help='Epochs for training: default=100')
    parser.add_argument("--learning_rate", type=float, default=0.0029, help='Learning rate for training: default=0.0029')
    parser.add_argument("--patience", type=int, default=10, help='Patience of the earlystopper: default=10')
    parser.add_argument("--verbose", type=int, default=100, help='Between batch range to print train accuracy: default=100')
    parser.add_argument("--threshold", type=float, default=0.0, help='Threshold used for predicting 0/1')
    parser.add_argument("--seed", type=int, default=227182, help='Seed used for reproduction')
    parser.add_argument("--fold_k", type=int, default=5, help='Number of fold for k-fold split. If k=1, standard train/val splitting is done.')
    parser.add_argument("--tta", dest='tta', action='store_true', help='Whether to use TTA on inference. Specify this argument to use TTA.')
    args = parser.parse_args()
    
    
    # ASSERT CONDITIONS
    assert (args.model_index > 0) and (args.mode in ['train', 'test'])
    
    
    LOG_PATH = os.path.join(args.base_dir, 'logs')
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s : %(message)s", 
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(os.path.join(LOG_PATH, f"log_model_{args.model_index}.txt")),
                            logging.StreamHandler()
                        ])
    logger.info("START")
    
    # ------------------------
    #   GLOBAL CUDA SETTING
    # ------------------------
    global_cuda = args.cuda and torch.cuda.is_available()
    if global_cuda:
        global_device = torch.device(f'cuda:{args.device_index}')
    else:
        global_device = torch.device('cpu')

    logger.info(f"Global Device: {global_device}")
    logger.info(f'Parsed Args: {args}')
    
    
    # ------------------
    #    Seed Setting
    # ------------------
    set_seed(args.seed)
    
    
    # -----------------------
    #      SET DIRECTORY
    # -----------------------
    base_dir = args.base_dir
    
    if args.data_type == 'original':
        data_to_use = ['trainset', 'testset']
    elif args.data_type == 'denoised':
        data_to_use = ['denoised_trainset_weak', 'denoised_testset_weak']
    else:
        raise Exception(f"Data Type Choice Error. No {args.data_type}")

    data_path_train = os.path.join('dataset', data_to_use[0]) 
    data_path_test = os.path.join('dataset', data_to_use[1])
    logger.info(f'Data used: train: {data_path_train}, test: {data_path_test}')
    
    img_dir_train = os.path.join(base_dir, data_path_train)
    img_dir_test = os.path.join(base_dir, data_path_test)

    label_dir_train = os.path.join(base_dir, 'dataset/dirty_mnist_2nd_answer.csv')
    label_dir_test = os.path.join(base_dir, 'dataset/sample_submission.csv')

    ckpt_folder_path = os.path.join(args.ckpt_path, f'model_{args.model_index}')
    
    
    
    # -------------------
    #   TRAIN/VAL SPLIT
    # -------------------
    if args.fold_k == 1:
        train_index_set, val_index_set = split_index(range(50000), args.val_ratio)
        train_index_set, val_index_set = [train_index_set], [val_index_set]
        
    elif args.fold_k > 1:
        splitted = split_kfold(args.fold_k, train_len=50000)
        train_index_set, val_index_set = [], []
        
        for train_fold, val_fold in splitted:
            train_index_set.append(train_fold)
            val_index_set.append(val_fold)
    
    test_index = range(5000)


    # ----------------
    #    Call model
    # ----------------
    base_model_type = args.base_model
    base_model = CallModel(model_type=base_model_type,
                           pretrained=args.pretrained,
                           logger=logger,
                           path=args.pretrained_weights_dir).model_return()
    
    model = base_model.to(global_device)
    
    
    # --------------------
    #        TRAIN
    # --------------------
    if args.mode == 'train':
        
        #  MAKE FOLDER for saving CHECKPOINTS
        # if folder already exists, assert. Else, make folder.
        assert not os.path.exists(ckpt_folder_path), "Model checkpoint folder already exists."
        os.makedirs(ckpt_folder_path)
        
        for k in range(args.fold_k):
            model_to_train = copy.deepcopy(model)
            
            logger.info(f"Training on Fold ({k+1}/{args.fold_k})")
            # Load trainset/valset
            train_index = train_index_set[k]
            val_index = val_index_set[k]
            
            train_loader, val_loader = load_dataset(mode='train',
                                                    batch_size=args.batch_size,
                                                    img_dir=img_dir_train,
                                                    label_dir=label_dir_train,
                                                    train_index=train_index,
                                                    val_index=val_index)
            
            # Train model
            train_model(model_to_train, k, ckpt_folder_path, args, logger, train_loader, val_loader)
    
    
    # --------------------
    #      INFERENCE
    # -------------------    
    if args.mode == 'test':
        
        test_loader = load_dataset(mode='test',
                                   img_dir=img_dir_test,
                                   label_dir=label_dir_test,
                                   test_index=test_index,
                                   tta = args.tta,
                                   angles = [0, 90, -90])
        
        pred_list = []
        for k in range(args.fold_k):
            logger.info(f"Inference using model of fold ({k+1}/{args.fold_k})")
            # Call model & trained weights
            model_inference = load_trained_weight(
                model_input=model,
                model_index=args.model_index,
                model_type='early',
                fold_k=k+1).to(global_device)
        
            pred = make_inference(args, model_inference, test_loader)
            pred_list.append(pred)
        
        # Aggregate and Save result
        aggregate_submit(args, pred_list)