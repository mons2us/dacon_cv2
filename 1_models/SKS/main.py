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

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from src.model import DaconModel, PretrainedResnet

from utils.imageprocess import image_transformer, image_processor
from utils.EarlyStopping import EarlyStopping
from utils.dataloader import CustomDataLoader
from utils.radams import RAdam

from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


def split_index(total_index, val_ratio):
    
    tot_len = len(total_index)
    train_len = int(tot_len*(1-val_ratio))

    train_sampled = random.sample(total_index, train_len)
    val_sampled = [i for i in total_index if i not in train_sampled]
    logger.info(f"Trainset length: {len(train_sampled)}, Valset length: {len(val_sampled)}")
    
    return train_sampled, val_sampled


def train_model(input_model, args, *loaders):
    
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    train_loader = loaders[0]
    val_loader = loaders[1]
    
    early_stopping = EarlyStopping(patience = args.patience, verbose = False)
    
    #loss_function = nn.MultiLabelSoftMarginLoss()
    loss_function = nn.BCEWithLogitsLoss()
    
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = RAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
        
    decayRate = 0.998
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10)
    
    train_tot_num = train_loader.dataset.__len__()
    val_tot_num = val_loader.dataset.__len__()
    
    train_corrects_sum = 0
    val_corrects_sum = 0
    
    logger.info(f'Training begins... Epochs = {epochs}')
    for epoch in range(epochs):
        
        print(f"\n-- Epoch: {str((epoch+1)).zfill(3)}th")
        
        torch.cuda.empty_cache()
        model.train()
        
        train_tmp_corrects_sum, train_tmp_num = 0, 0
        for idx, (train_X, train_Y) in enumerate(train_loader):

            train_tmp_num += len(train_Y)
            
            optimizer.zero_grad()
            train_pred = model(train_X)
            
            train_loss = loss_function(train_pred, train_Y)
            train_loss.backward()
            optimizer.step()
            
            train_pred_label = train_pred >= 0.5
            train_corrects = (train_pred_label == train_Y).sum()
            
            train_corrects_sum += train_corrects.item()
            train_tmp_corrects_sum += train_corrects.item()
            
            # Check between batches
            verbose = args.verbose
            
            if (idx+1) % verbose == 0:
                print(f"------- {str((idx+1)).zfill(4)}th batch | Train Acc: {train_tmp_corrects_sum/(train_tmp_num*26)*100:.4f}%")
                
                # initialization
                train_tmp_corrects_sum, train_tmp_num = 0, 0
            
        
        with torch.no_grad():
            
            for idx, (val_X, val_Y) in enumerate(val_loader):
                
                val_pred = model(val_X)
                val_loss = loss_function(val_pred, val_Y)
                
                val_pred_label = val_pred >= 0.5
                val_corrects = (val_pred_label == val_Y).sum()
                val_corrects_sum += val_corrects.item()
                
        train_acc = train_corrects_sum/(train_tot_num*26)*100
        val_acc = val_corrects_sum/(val_tot_num*26)*100
        
        print(f"-- Epoch: {str((epoch+1)).zfill(3)}th finished --- Training Acc: {train_acc:.4f}% | Val Acc: {val_acc:.4f}%")
        
        train_corrects_sum = 0
        val_corrects_sum = 0
        
        early_stopping(-val_acc, model)
        
        if early_stopping.early_stop:
            print("Early stopping condition met --- TRAINING STOPPED")
            break
        
        lr_scheduler.step()
    
    return model


if __name__ == "__main__":
    
    logger.info("START")
    
    # ARGUMENTS PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/home/sks/COMPETITION/DACON/computer_vision2", help='Base PATH of your work')
    parser.add_argument("--weights_path", type=str, default="/home/sks/COMPETITION/DACON/computer_vision2/pretrained_model", help='PATH to weights of pretrained model')
    parser.add_argument("--cuda", dest='cuda', action='store_false', help='Whether to use CUDA: defuault is True, so specify this argument not to use CUDA')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size for train-loader for training phase')
    parser.add_argument("--val_ratio", type=float, default=0.1, help='Ratio for validation set: default=0.1')
    parser.add_argument("--epochs", type=int, default=100, help='Epochs for training: default=100')
    parser.add_argument("--learning_rate", type=float, default=0.0029, help='Learning rate for training: default=0.005')
    parser.add_argument("--patience", type=int, default=3, help='Patience of the earlystopper: default=3')
    parser.add_argument("--verbose", type=int, default=100, help='Between batch range to print train accuracy: default=100')
    args = parser.parse_args()
    
    
    # GLOBAL CUDA SETTING
    global_cuda = args.cuda
    global_device = torch.device("cuda:0") if global_cuda and torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Global Device: {global_device}")
    
    
    # SET DIRECTORY
    base_dir = args.base_dir
    img_dir_train = os.path.join(base_dir, 'dataset/denoised_trainset')
    img_dir_test = os.path.join(base_dir, 'dataset/denoised_testset')

    label_dir_train = os.path.join(base_dir, 'dataset/dirty_mnist_2nd_answer.csv')
    label_dir_test = os.path.join(base_dir, 'dataset/sample_submission.csv')
    
    
    # TRAIN/VAL SPLIT
    train_index, val_index = split_index(range(50000), args.val_ratio)
    test_index = range(5000)
    
    # LOAD DATASET
    train_set = CustomDataLoader(img_dir=img_dir_train,
                                  label_dir=label_dir_train,
                                  train=True,
                                  row_index=train_index,
                                  device=global_device)
    
    val_set = CustomDataLoader(img_dir=img_dir_train,
                               label_dir=label_dir_train,
                               train=False,
                               row_index=val_index,
                               device=global_device)
    
    test_set = CustomDataLoader(img_dir=img_dir_test,
                                label_dir=label_dir_test,
                                train=False,
                                row_index=test_index,
                                device=global_device)
    
    #train_set, val_set = torch.utils.data.random_split(train_base, lengths=[42000, 8000])
    
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32)
    
    
    # CALL MODEL
    # model = DaconModel().to(global_device)
    resnet_model = resnet50()
    resnet_model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.Linear(256, 26)
    )
    
    model = PretrainedResnet(resnet_model).to(global_device)

    # load weights
    pretrained_weights = os.path.join(args.weights_path, 'pretrained_resnet.pth')
    model.load_state_dict(torch.load(pretrained_weights))
    
    # CHANGE LAST LAYER ((256, 26) -> (256, 26))
    model.block[2].fc[1] = nn.Sequential(
        nn.ReLU(),
        nn.Linear(256, 26)
    ).to(global_device)

    # --------------------
    # TRAIN
    #
    # parameters:
    # `epochs`
    # `learning_rate`
    # -------------------
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    train_model(model, args, train_loader, val_loader)
    
    