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
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import math

from sklearn.model_selection import train_test_split


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

parent_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.append(parent_dir)
sys.path.append(os.path.abspath(os.path.join(parent_dir, '..')))
    
from utilities import *


from torch.optim import lr_scheduler


class DecoderDropout(nn.Module):
    def __init__(self, inp_dim, num_v_genes, num_j_genes, hidden_layer_sizes=[1024, 512], activation_f=nn.ReLU, n_drop=0.25, use_j=False):
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
        self.fc_j = nn.Linear(inp_layer_size, num_j_genes)
        
        if use_j:
            self.fc_v_input = nn.Linear(inp_layer_size + num_j_genes, inp_layer_size)
        
        self.fc_v = nn.Linear(inp_layer_size, num_v_genes)
               
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, use_j=False):
        x = self.hidden_net(x)
        j_gene = self.softmax(self.fc_j(x))  
        
        if not use_j:
            v_gene = self.softmax(self.fc_v(x))
            
        else:
            v_preds_j = torch.cat([x, j_gene], dim=1)
            v_preds_j = self.fc_v_input(v_preds_j)
            v_gene = self.softmax(self.fc_v(v_preds_j))
        
        return v_gene, j_gene
    


class DecoderModelDropout():#MLPClassifier):
    def __init__(self, n_v, n_j, hidden_layer_sizes=[1024, 512], activation_f=nn.ReLU, lr=0.0001, n_drop=0.25, use_j=False):
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
        self.use_j = use_j
        
    def fit(self, data, labels, n_epochs=15, criterion_v = nn.CrossEntropyLoss(), criterion_j = nn.CrossEntropyLoss(), filename="info_losses.txt"):
        inp_dim = data.shape[1]
        self.model = DecoderDropout(inp_dim=inp_dim, num_v_genes=self.n_v, num_j_genes=self.n_j, hidden_layer_sizes=self.hidden_layer_sizes, activation_f=self.activation_f, n_drop=self.n_drop, use_j=self.use_j)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

        
        losses = []
        losses_v = []
        losses_j = []

        X = torch.tensor(data.values, dtype=torch.float32)
        y = torch.tensor(labels.values, dtype=torch.long).reshape(-1, 2)
        data_loader = DataLoader(list(zip(X, y)), batch_size=32)
        output_file = open(filename, "a")

        for epoch in range(n_epochs):
            epoch_loss = 0
            v_loss_ = 0
            j_loss_ = 0

            self.model.train()

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
            
            scheduler.step(epoch_loss)

            # print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            output_file.write(f'Epoch {epoch+1}, Loss: {loss.item()}\n')
            
        self.loss = losses
        self.loss_v = losses_v
        self.loss_j = losses_j
        output_file.close()

        
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

    
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)

# b_dist = pd.read_csv('tcremp_b/tcremp_dists_TRB.txt', sep='\t', low_memory=False)
b_dist = pd.read_csv('../tcremp_b/tcremp_dists_TRB.txt', sep='\t', low_memory=False)
b_dist = b_dist.drop(columns = ['tcremp_id', 'cloneId'])

b_dist.b_v = b_dist.b_v.apply(process_types)
b_dist.b_j = b_dist.b_j.apply(process_types)

bv_minor = b_dist.b_v.value_counts()[b_dist.b_v.value_counts() < 200].index
bj_minor = b_dist.b_j.value_counts()[b_dist.b_j.value_counts() < 200].index

b_v_enc = LabelEncoder()
b_j_enc = LabelEncoder()

b_dist_procssed = preprocess_df(b_dist, 'b_v', 'b_j', b_v_enc, b_j_enc, bv_minor, bj_minor)

b_train, b_test = train_test_split(b_dist_procssed, test_size=0.2)

b_train = b_train.drop(columns=['b_cdr3aa'])

n_v = b_dist_procssed.b_v.nunique()
n_j = b_dist_procssed.b_j.nunique()
inp_dim = b_train.shape[1] - 2

X = b_train.iloc[:, 2:]
y = b_train.iloc[:, 0:2]

b_test = b_test.drop(columns=['b_cdr3aa'])
X_test = b_test.iloc[:, 2:]
y_test = b_test.iloc[:, 0:2]

hid_layer_size = [[2048, 1024, 512, 256], [1024, 512], [1024, 512, 256], [1024, 512, 256, 128], [1024, 512, 320, 180], [1500, 1000, 750, 500, 200]]
l_rs = [0.0001, 0.00001]
n_drop = [0.2, 0.3, 0.4]
use_j = [True, False]

n_epochs=30
with_drop=True
    
# def model_choice(hid_layer_size, l_rs, n_epochs, with_drop=False, n_drop=[0.25], filename="info_losses.txt"):

scores = {}
losses = {}
if with_drop:
    n_iters = len(hid_layer_size)*len(l_rs)*len(n_drop)*len(use_j)
else:
    n_iters = len(hid_layer_size)*len(l_rs)*len(use_j)
curr_iter = 1
for size_i in range(len(hid_layer_size)):
    for rate_i in range(len(l_rs)):
        for drop_i in range(len(n_drop)):
            for j_use_i in use_j:
                    # print
                output_file = open("info_losses.txt", "a")
                output_file.write(f'Iteration {curr_iter} of {n_iters}\n')
                curr_iter+=1
                size = hid_layer_size[size_i]
                rate = l_rs[rate_i]
                drop = n_drop[drop_i]
                j_use = j_use_i
                if with_drop:
                    curr_params = f'Hid_layer_size: {size}, lr: {rate}, n_drop: {drop}, use j: {j_use}'
                    model = DecoderModelDropout(n_v=n_v, n_j=n_j, hidden_layer_sizes=size, lr=rate, n_drop=drop, use_j=j_use)

                else:
                    curr_params = f'Hid_layer_size: {size}, lr: {rate}, use j {j_use}'
                    model = DecoderModel(n_v=n_v, n_j=n_j, hidden_layer_sizes=size, lr=rate, use_j=j_use)

                output_file.write(f'   ~ Training for parameters: {curr_params} ~\n')
                output_file.close()
                
                model.fit(X, y, n_epochs=n_epochs, filename="info_losses.txt")
                losses[curr_params] = model.loss

                v_true, j_true = y_test.values.T
                v_preds, j_preds = model.predict(X_test)
                v_accuracy = accuracy_score(v_true, v_preds)
                j_accuracy = accuracy_score(j_true, j_preds)

                scores[curr_params] = (v_accuracy, j_accuracy)
                    # print
                output_file = open("info_losses.txt", "a")
                output_file.write(f' ~ Training finished ~\n Score: V accuracy:{v_accuracy}; J accuracy: {j_accuracy}\n')
                output_file.close()
            
epochs = range(1, n_epochs + 1)

    
best_params = sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)[0]
output_file.write(f'Best parameters found: {best_params[0]}\nV accuracy:{best_params[1][0]}\nJ accuracy:{best_params[1][1]}')
    
losses_df = pd.DataFrame.from_dict(losses)
scores_df = pd.DataFrame.from_dict(scores)

losses_df.to_csv('losses.csv', index=False)
scores_df.to_csv('scores.csv', index=False)    


