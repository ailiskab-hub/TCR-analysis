import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
import torch
from tqdm import tqdm
import os
import sys
from sklearn.utils import resample
from datetime import datetime
import math

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def save_model(model, filename='model_params.pth'):
    torch.save(model.model.state_dict(), filename)
    
def load_trained_model(inp_dim, n_v, n_j, hidden_layer_sizes=[1024, 512], lr=0.0001, activation_f=nn.ReLU, n_drop=0.25, filename='model_params.pth'):
    loaded_model = DecoderModelDropout(n_v=n_v, n_j=n_j, hidden_layer_sizes=hidden_layer_sizes, lr=lr, n_drop=n_drop)
    loaded_model.model = DecoderDropout(inp_dim=inp_dim, num_v_genes=n_v, num_j_genes=n_j, hidden_layer_sizes=hidden_layer_sizes, activation_f=activation_f, n_drop=n_drop)
    loaded_model.model.load_state_dict(torch.load(filename))
    
    return loaded_model



class DecoderDropout(nn.Module):
    def __init__(self, inp_dim, num_v_genes, num_j_genes, hidden_layer_sizes=[1024, 512], activation_f=nn.ReLU, n_drop=0.25):
        super(DecoderDropout, self).__init__()
        layers = []
        inp_layer_size = inp_dim
        self.dropout = nn.Dropout(n_drop)
        ind = 1
        for layer_size in hidden_layer_sizes:
            layers.append(nn.Linear(inp_layer_size, layer_size))
            layers.append(activation_f())
            if ind % 2 != 0:
                layers.append(nn.Dropout(n_drop))
            inp_layer_size = layer_size
            ind += 1
        
        self.hidden_net = nn.Sequential(*layers)
        self.fc_v = nn.Linear(inp_layer_size, num_v_genes)
        self.fc_j = nn.Linear(inp_layer_size, num_j_genes)
        
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.hidden_net(x)
        v_gene = self.softmax(self.fc_v(x))
        j_gene = self.softmax(self.fc_j(x))  
        
        return v_gene, j_gene
    
    

class DecoderModelDropout():#MLPClassifier):
    def __init__(self, n_v, n_j, hidden_layer_sizes=[1024, 512], activation_f=nn.ReLU, lr=0.0001, n_drop=0.25):
        # super(DecoderModel, self).__init__()
        self.n_v = n_v
        self.n_j = n_j
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_f = activation_f
        self.lr = lr
        self.loss = []
        self.loss_v = []
        self.loss_j = []
        self.model = None
        self.n_drop = n_drop
        
    def fit(self, data, labels, n_epochs=15, criterion_v = nn.CrossEntropyLoss(), criterion_j = nn.CrossEntropyLoss()):
        inp_dim = data.shape[1]
        self.model = DecoderDropout(inp_dim=inp_dim, num_v_genes=self.n_v, num_j_genes=self.n_j, hidden_layer_sizes=self.hidden_layer_sizes, activation_f=self.activation_f, n_drop=self.n_drop)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        losses = []
        losses_v = []
        losses_j = []

        X = torch.tensor(data.values, dtype=torch.float32)
        y = torch.tensor(labels.values, dtype=torch.long).reshape(-1, 2)
        data_loader = DataLoader(list(zip(X, y)), batch_size=32)
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            v_loss_ = 0
            j_loss_ = 0

            for data in data_loader:
                # print(data)
                x, vj = data
                v_target, j_target = vj.T

                optimizer.zero_grad()

                v_pred, j_pred = self.model(x)
                
                loss_v = criterion_v(v_pred, v_target)
                loss_j = criterion_j(j_pred, j_target)

                v_loss_ += loss_v.item()
                j_loss_ += loss_j.item()

                loss = loss_v + loss_j

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss / len(data_loader))
            losses_v.append(v_loss_ / len(data_loader))
            losses_j.append(j_loss_ / len(data_loader))

            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            
        self.loss = losses
        self.loss_v = losses_v
        self.loss_j = losses_j


    def predict(self, data_x):
        data_x = torch.tensor(data_x.values, dtype=torch.float32)
        data_loader = DataLoader(data_x, batch_size=32)
        self.model.eval()
        
        v_preds = []
        j_preds = []
        
        with torch.no_grad():  
            for x in data_loader:
                # print(x)

                v_pred, j_pred = self.model(x)

                v_pred_ = v_pred.argmax(dim=1)
                j_pred_ = j_pred.argmax(dim=1)

                v_preds.extend(v_pred_.cpu().numpy())
                j_preds.extend(j_pred_.cpu().numpy())


        return v_preds, j_preds

    
class DecoderDropoutV(nn.Module):
    def __init__(self, inp_dim, num_v_genes, hidden_layer_sizes=[1024, 512], activation_f=nn.ReLU, n_drop=0.25):
        super(DecoderDropoutV, self).__init__()
        layers = []
        inp_layer_size = inp_dim
        self.dropout = nn.Dropout(n_drop)
        ind = 1
        for layer_size in hidden_layer_sizes:
            layers.append(nn.Linear(inp_layer_size, layer_size))
            layers.append(activation_f())
            if ind % 2 != 0:
                layers.append(nn.Dropout(n_drop))
            inp_layer_size = layer_size
            ind += 1
        
        self.hidden_net = nn.Sequential(*layers)
        self.fc_v = nn.Linear(inp_layer_size, num_v_genes)        
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.hidden_net(x)
        v_gene = self.softmax(self.fc_v(x))
        
        return v_gene
    
    

class DecoderModelDropoutV():#MLPClassifier):
    def __init__(self, n_v, hidden_layer_sizes=[1024, 512], activation_f=nn.ReLU, lr=0.0001, n_drop=0.25):
        self.n_v = n_v
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_f = activation_f
        self.lr = lr
        self.loss = []
        self.loss_v = []
        self.model = None
        self.n_drop = n_drop
        
    def fit(self, data, labels, n_epochs=15, criterion_v = nn.CrossEntropyLoss()):
        inp_dim = data.shape[1]
        self.model = DecoderDropoutV(inp_dim=inp_dim, num_v_genes=self.n_v, hidden_layer_sizes=self.hidden_layer_sizes, activation_f=self.activation_f, n_drop=self.n_drop)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        losses = []
        losses_v = []

        X = torch.tensor(data.values, dtype=torch.float32)
        y = torch.tensor(labels.values, dtype=torch.long)#.reshape(-1, 2)
        data_loader = DataLoader(list(zip(X, y)), batch_size=32)
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            v_loss_ = 0

            for data in data_loader:
                # print(data)
                x, v_target = data

                optimizer.zero_grad()

                v_pred = self.model(x)
                loss_v = criterion_v(v_pred, v_target)

                loss = loss_v

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss / len(data_loader))

            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            
        self.loss = losses


    def predict(self, data_x):
        data_x = torch.tensor(data_x.values, dtype=torch.float32)
        data_loader = DataLoader(data_x, batch_size=32)
        self.model.eval()
        
        v_preds = []
        
        with torch.no_grad():  
            for x in data_loader:
                v_pred = self.model(x)

                v_pred_ = v_pred.argmax(dim=1)

                v_preds.extend(v_pred_.cpu().numpy())


        return v_preds
