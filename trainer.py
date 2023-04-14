import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from utils.utils import EarlyStopping
import numpy as np


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.optimizer = config['optimizer']
        self.loss_function = config['loss_function']
        self.train_dl = config['train_dataloader']
        self.val_dl = config['val_dataloader']
        self.test_dl = config['test_dataloader']
        self.epochs = config['epochs']
        self.device = config['device']
        self.log_dict = {
            'training_loss_per_batch': [],
            'validation_loss_per_batch': [],
            'training_accuracy_per_epoch': [],
            'validation_accuracy_per_epoch': []
        }
    
    def accuracy(self, model, dataloader):
        model.eval()
        total_correct = 0
        total_instances = 0
        
        for datas, labels in tqdm(dataloader):
            datas, labels = datas.to(self.device), labels.to(self.device)
            predictions = torch.argmax(model(datas), dim=1)
            correct_predictions = sum(predictions==labels).item()
            total_correct+=correct_predictions
            total_instances+=len(datas)
            
        return round(total_correct/total_instances, 3)
    
        
    def train(self):
        early_stopping = EarlyStopping(patience = 20, verbose = True, path='./models/weights/checkpoint.pt')
        
        for epoch in range(1, self.epochs + 1):
            train_losses = []
            val_losses = []
            train_accuracy = 0.0
            loss_total = 0.0
            
            self.model.train()
            
            for train_datas, traon_labels in tqdm(self.train_dl, desc='train'):
                train_datas, traon_labels = train_datas.to(self.device), traon_labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(train_datas)
                train_loss = self.loss_function(outputs, traon_labels)
                self.log_dict['training_loss_per_batch'].append(train_loss.item())
                train_losses.append(train_loss.item())
                train_loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                # computing training accuracy
                train_accuracy = self.accuracy(self.model, self.train_dl)
                self.log_dict['training_accuracy_per_epoch'].append(train_accuracy)
                
            
            print("validating ... ")
            self.model.eval()
            
            with torch.no_grad():
                for val_datas, val_labels in tqdm(self.val_dl, desc='validation'):
                    val_datas, val_labels = val_datas.to(self.device), val_labels.to(self.device)
                    outputs = self.model(val_datas)
                    val_loss = self.loss_function(outputs, val_labels)
                    self.log_dict['validation_loss_per_batch'].append(val_loss.item())
                    val_losses.append(val_loss.item())
                # computing accuracy
                val_accuracy = self.accuracy(self.model, self.val_dl)
                self.log_dict['validation_accuracy_per_epoch'].append(val_accuracy)
            
            train_losses = np.array(train_losses).mean()
            val_losses = np.array(train_losses).mean()
            
            early_stopping(val_loss, self.model)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break
        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(train_losses)+1), train_losses, label='Training Loss')
        plt.plot(range(1,len(val_losses)+1),val_losses,label='Validation Loss')
        plt.plot(range(len(self.log_dict['training_accuracy_per_epoch'])), self.log_dict['training_accuracy_per_epoch'], label='Training Accuracy')
        plt.plot(range(len(self.log_dict['validation_accuracy_per_epoch'])), self.log_dict['validation_accuracy_per_epoch'], label='Validation Accuracy')
        plt.legend()
        plt.show()
                
                
            

                
        
        
        