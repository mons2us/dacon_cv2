import os
import csv
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from src.model import DaconModel

from utils.imageprocess import image_transformer, image_processor
from utils.EarlyStopping import EarlyStopping
from utils.dataloader import CustomDataLoader

from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


def train_model(input_model, epochs, lr, *loaders):
    
    train_loader = loaders[0]
    val_loader = loaders[1]
    
    early_stopping = EarlyStopping(patience = 5, verbose = False)
    
    loss_function = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    decayRate = 0.998
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    
    train_tot_num = train_loader.dataset.__len__()
    val_tot_num = val_loader.dataset.__len__()
    
    train_corrects_sum = 0.0
    val_corrects_sum = 0.0
    
    logger.info(f'Training begins... Epochs = {epochs}')
    for epoch in range(epochs):
        
        torch.cuda.empty_cache()
        model.train()
        
        for idx, (train_X, train_Y) in enumerate(tqdm(train_loader, desc="Training phase per batch")):
            
            optimizer.zero_grad()
            train_pred = model(train_X)
            
            train_loss = loss_function(train_pred, train_Y)
            train_loss.backward()
            optimizer.step()
            
            train_pred_label = train_pred >= 0.5
            train_corrects = (train_pred_label == train_Y).sum()
            train_corrects_sum += train_corrects.item()
        
        with torch.no_grad():
            
            for idx, (val_X, val_Y) in enumerate(val_loader):
                
                val_pred = moedl(val_X)
                val_loss = loss_function(val_pred, val_Y)
                
                val_pred_label = val_pred >= 0.5
                val_corrects = (val_pred_label == val_Y).sum()
                val_corrects_sum += val_corrects.item()
                
        train_acc = train_corrects_sum / train_tot_num * 100
        val_acc = val_corrects_sum / val_tot_num * 100
        
        print(f"Epoch: {epoch+1} | Training Acc: {train_acc:.4f}% | Val Acc: {val_acc:.4f}%")
        
        train_corrects_sum = 0
        val_corrects_sum = 0
        
        early_stopping(-val_accuracy, model)
        
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
    parser.add_argument("--cuda", dest='cuda', action='store_false', help='Whether to use CUDA: defuault is True, so specify this argument not to use CUDA')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size for train-loader for training phase')
    args = parser.parse_args()
    
    # GLOBAL CUDA SETTING
    global_cuda = args.cuda
    global_device = torch.device("cuda:0") if global_cuda and torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Global Device: {global_device})
    
    
    # SET DIRECTORY
    base_dir = args.base_dir
    img_dir_train = os.path.join(base_dir, 'dataset/trainset')
    img_dir_test = os.path.join(base_dir, 'dataset/testset')

    label_dir_train = os.path.join(base_dir, 'dataset/dirty_mnist_2nd_answer.csv')
    label_dir_test = os.path.join(base_dir, 'dataset/sample_submission.csv')
    
    # LOAD DATASET
    train_base = CustomDataLoader(img_dir_train, label_dir_train, True, global_device)
    test_set = CustomDataLoader(img_dir_test, label_dir_test, True, global_device)

    train_set, val_set = torch.utils.data.random_split(train_base, lengths=[45000, 5000])
    
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32)
    
    
    # CALL MODEL
    model = DaconModel().to(global_device)
    
    # --------------------
    # TRAIN
    #
    # parameters:
    # `epochs`
    # `learning_rate`
    # -------------------
    epochs = 50
    learning_rate = 0.0025
    
    train_model(model, epochs, learning_rate, train_loader, val_loader)
    
    