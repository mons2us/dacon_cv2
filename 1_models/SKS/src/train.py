import os
import csv
import cv2
import numpy as np
from time import time

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from warmup_scheduler import GradualWarmupScheduler

from src.model import DaconModel, PretrainedResnet
from utils.imageprocess import image_transformer, image_processor
from utils.EarlyStopping import EarlyStopping
from utils.dataloader import CustomDataLoader
from utils.radams import RAdam

from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


def train_model(input_model, args, *loaders):
    
    model = input_model
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    train_loader = loaders[0]
    val_loader = loaders[1]
    
    early_stopping = EarlyStopping(patience = args.patience, verbose = False)
    
    loss_function = nn.MultiLabelSoftMarginLoss()
    #loss_function = nn.BCEWithLogitsLoss()
    #loss_function = nn.BCELoss()
    
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = RAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    
    
    # LERANING RATE SCHEDULER (WARMUP)
    #decayRate = 0.998
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=25)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.5)
    lr_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=int(args.epochs*0.1), after_scheduler=lr_scheduler)
    
    train_tot_num = train_loader.dataset.__len__()
    val_tot_num = val_loader.dataset.__len__()
    
    train_corrects_sum = 0
    train_loss_sum = 0.0
    
    val_corrects_sum = 0
    val_loss_sum = 0.0
    
    logger.info(f'Training begins... Epochs = {epochs}')
    
    for epoch in range(epochs):

        time_start = time()
        
        # warmup scheduler step / lr scheduler
        lr_warmup.step()
        
        print(f"""
    ---------------------------------------------------------------------------
        Phase INFO
            Current phase : {epoch+1}th epoch
            Learning Rate : {optimizer.param_groups[0]['lr']:.6f}
    ---------------------------------------------------------------------------""", end='')
        
        torch.cuda.empty_cache()
        model.train()
        
        train_tmp_num = 0
        train_tmp_corrects_sum = 0
        train_tmp_loss_sum = 0.0
        for idx, (train_X, train_Y) in enumerate(train_loader):

            train_tmp_num += len(train_Y)
            
            optimizer.zero_grad()
            train_pred = model(train_X)
            train_loss = loss_function(train_pred, train_Y)
            train_loss.backward()
            optimizer.step()
            
            train_pred_label = train_pred >= args.threshold
            train_corrects = (train_pred_label == train_Y).sum()
            
            train_corrects_sum += train_corrects.item()
            train_loss_sum += train_loss.item()
            
            train_tmp_corrects_sum += train_corrects.item()
            train_tmp_loss_sum += train_loss.item()
            
            # Check between batches
            verbose = args.verbose
            
            if (idx+1) % verbose == 0:
                print(f"""
    -- ({str((idx+1)).zfill(4)} / {str(len(train_loader)).zfill(4)}) Train Loss: {train_tmp_loss_sum/train_tmp_num:.6f} | Train Acc: {train_tmp_corrects_sum/(train_tmp_num*26)*100:.4f}%""", end='')
                
                # initialization
                train_tmp_num = 0
                train_tmp_corrects_sum = 0
                train_tmp_loss_sum = 0.0
            
        
        with torch.no_grad():
            
            for idx, (val_X, val_Y) in enumerate(val_loader):
                
                val_pred = model(val_X)
                val_loss = loss_function(val_pred, val_Y)
                
                val_pred_label = val_pred >= args.threshold
                val_corrects = (val_pred_label == val_Y).sum()
                val_corrects_sum += val_corrects.item()
                
                val_loss_sum += val_loss.item()
                
        train_acc = train_corrects_sum/(train_tot_num*26)*100
        train_loss = train_loss_sum/train_tot_num
        
        val_acc = val_corrects_sum/(val_tot_num*26)*100
        val_loss = val_loss_sum/val_tot_num
        
        time_end = time()
        time_len_m, time_len_s = divmod(time_end - time_start, 60)
        
        print(f"""
    ---------------------------------------------------------------------------
        Phase SUMMARY
            Finished phase  : {epoch+1}th epoch
            Time taken      : {time_len_m:.0f}m {time_len_s:.2f}s
            Training Loss   : {train_loss:.6f}  |  Training Acc   : {train_acc:.4f}%
            Validation Loss : {val_loss:.6f}  |  Validation Acc : {val_acc:.4f}%
    ---------------------------------------------------------------------------
        """)
        
        if ((epoch+1) >= 10) and ((epoch+1) % 5 == 0):
            save_path = os.path.join(args.base_dir, f'checkpoints/model_ckpt_{epoch+1}.pth')
            logger.info(f"SAVING MODEL: {save_path}")
            torch.save(model.state_dict(), save_path)
        
        # INITIALIZATION
        train_corrects_sum = 0
        val_corrects_sum = 0
        
        train_loss_sum = 0.0
        val_loss_sum = 0.0
        
        # EARLY STOPPER
        early_stopping(-val_acc, model)
        
        if early_stopping.early_stop:
            print("Early stopping condition met --- TRAINING STOPPED")
            break
            
        # lr scheduler step
        lr_scheduler.step()

    return model