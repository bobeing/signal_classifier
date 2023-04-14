import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from models.model import Model

from torchvision import utils
import matplotlib.pyplot as plt
from utils.utils import EarlyStopping

from torchsummary import summary
import time
import os
import json
from tqdm import tqdm
import csv
import h5py
import numpy as np

import torchvision.transforms as transforms
from torchvision import models

import sys
# sys.path.append(r'D:\company_project\SOLETOP\for_gitlab\sigint\signal_classifier\calculation')
from utils.utils import read_h5_file, calc_stft

from torch.utils.data import Dataset
from trainer import Trainer

class CustomDataset(Dataset):
    def __init__(self, csv_path):
        self.signal_list = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.signal_list = list(reader)
        
    def __len__(self):
        return len(self.signal_list)
    
    def __getitem__(self, index):
        h5_data = self.signal_list[index]
        train_sig_attrs, train_sig = read_h5_file(h5_data[0])
        signal_img_data = calc_stft(train_sig, int(train_sig_attrs["samprate"]), 128)
        
        label = int(h5_data[1])
        
        return np.expand_dims(signal_img_data, axis=0), label

if __name__ == "__main__":
    
    patience = 10
    epochs = 50
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().resnet50().to(device)
    
    modes = ["./dataset/train.csv",
             "./dataset/val.csv",
             "./dataset/test.csv"]
    
    train_dataset = CustomDataset(csv_path = modes[0])
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    val_dataset = CustomDataset(csv_path = modes[1])
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    
    test_dataset = CustomDataset(csv_path = modes[2])
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    params = {
        'epochs':epochs,
        'optimizer':optimizer,
        'loss_function':loss_function,
        'train_dataloader':train_dataloader,
        'val_dataloader':val_dataloader,
        'test_dataloader': test_dataloader,
        'device':device
        }
    
    trainer = Trainer(model, params)
    trainer.train()
    
   