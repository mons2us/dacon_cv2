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

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from warmup_scheduler import GradualWarmupScheduler


from src.model import DaconModel, PretrainedResnet
from src.train import train_model
from utils.imageprocess import image_transformer, image_processor
from utils.EarlyStopping import EarlyStopping
from utils.dataloader import CustomDataLoader
from utils.radams import RAdam

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    
    # ARGUMENTS PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_index", type=int, default=0, help='My model index. Integer type, and should be greater than 0')
    parser.add_argument("--base_dir", type=str, default="/home/sks/COMPETITION/DACON/computer_vision2", help='Base PATH of your work')
    parser.add_argument("--mode", type=str, default="train", help='[train | test]')
    parser.add_argument("--data_type", type=str, default="denoised", help='[original | denoised]: default=denoised')
    parser.add_argument("--ckpt_path", type=str, default="/home/sks/COMPETITION/DACON/computer_vision2/checkpoints", help='PATH to weights of ckpts.')
    parser.add_argument("--pretrained_weights_path", type=str, default="/home/sks/COMPETITION/DACON/computer_vision2/pretrained_model", help='PATH to weights of pretrained model')
    parser.add_argument("--cuda", dest='cuda', action='store_false', help='Whether to use CUDA: defuault is True, so specify this argument not to use CUDA')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size for train-loader for training phase')
    parser.add_argument("--val_ratio", type=float, default=0.1, help='Ratio for validation set: default=0.1')
    parser.add_argument("--epochs", type=int, default=100, help='Epochs for training: default=100')
    parser.add_argument("--learning_rate", type=float, default=0.0029, help='Learning rate for training: default=0.0029')
    parser.add_argument("--patience", type=int, default=10, help='Patience of the earlystopper: default=10')
    parser.add_argument("--verbose", type=int, default=100, help='Between batch range to print train accuracy: default=100')
    parser.add_argument("--threshold", type=float, default=0.5, help='Threshold used for predicting 0/1')
    parser.add_argument("--seed", type=int, default=2020011135, help='Seed used for reproduction')
    args = parser.parse_args()
    
    
    # ASSERT CONDITIONS
    assert (args.model_index > 0) and (args.mode in ['train', 'test'])
    
    logger = logging.getLogger(__name__)
    log_path = os.path.join(args.base_dir, f'logs/log_model_{args.model_index}.txt')
    logging.basicConfig(filename=log_path, format='%(asctime)s : %(message)s', level=logging.INFO)

    logger.info("START")
    
    # GLOBAL CUDA SETTING
    global_cuda = args.cuda
    global_device = torch.device("cuda:0") if global_cuda and torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Global Device: {global_device}")
    
    logger.info(f'Parsed Args: {args}')
    
    
    # SET DIRECTORY
    base_dir = args.base_dir
    data_type = ['denoised_trainset_weak', 'denoised_testset_weak'] if args.data_type=='denoised' else ['trainset', 'testset']
    data_path_train, data_path_test = os.path.join('dataset', data_type[0]), os.path.join('dataset', data_type[1])
    logger.info(f'Data used: train: {data_path_train}, test: {data_path_test}')
    
    img_dir_train = os.path.join(base_dir, data_path_train)
    img_dir_test = os.path.join(base_dir, data_path_test)

    label_dir_train = os.path.join(base_dir, 'dataset/dirty_mnist_2nd_answer.csv')
    label_dir_test = os.path.join(base_dir, 'dataset/sample_submission.csv')
    
    
    # Seed setting
    set_seed(args.seed)
    
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
        nn.BatchNorm1d(256),
        
        nn.Linear(256, 26),
    )
    
    model_train = PretrainedResnet(resnet_model).to(global_device)

    # load weights
    pretrained_weights = os.path.join(args.pretrained_weights_path, 'pretrained_resnet.pth')
    model_train.load_state_dict(torch.load(pretrained_weights))
    
    # CHANGE LAST LAYER ((256, 26) -> (256, 26))
#     model.block[2].fc[1] = nn.Sequential(
#         nn.BatchNorm1d(256),
#         nn.Linear(256, 26),
#     ).to(global_device)


    epochs = args.epochs
    learning_rate = args.learning_rate
    
    # --------------------
    # TRAIN
    #
    # parameters:
    # `epochs`
    # `learning_rate`
    # -------------------    
    if args.mode == 'train':
        train_model(model_train, args, train_loader, val_loader)
    
    # --------------------
    # INFERENCE
    # -------------------    
    if args.mode == 'test':
        
        submission_file = pd.read_csv(os.path.join(args.base_dir, 'dataset/sample_submission.csv'))
        
        # CALL MODEL
        model_inference = PretrainedResnet(resnet_model).to(global_device)

        ### CHANGE LAST LAYER ((256, 26) -> (256, 26))
#         model.block[2].fc[1] = nn.Sequential(
#             nn.BatchNorm1d(256),
#             nn.Linear(256, 26),
#         ).to(global_device)

        # load weights
        ckpt_weights = os.path.join(args.ckpt_path, 'earlystopped_model.pth')
        model_inference.load_state_dict(torch.load(ckpt_weights))
        
        model_inference.eval()
        
        fin_labels = []
        for idx in tqdm(test_index):
            to_test = test_set.__getitem__(idx)[0].unsqueeze(0)
            
            pred = model_inference(to_test)
            pred_label = ((pred >= args.threshold)*1).detach().to('cpu').numpy()[0]
            fin_labels.append(pred_label)
        
        fin_result = np.array(fin_labels)
        submission_file.iloc[:, 1:] = fin_result
        
        save_path = os.path.join(args.base_dir, f'submit/submission_model_{args.model_index}.csv')
        submission_file.to_csv(save_path, index=False)