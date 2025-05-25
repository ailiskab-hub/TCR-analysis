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
# from datetime import datetime
# from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import math

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# from models import *

def process_types(string):
    return string.split('*')[0]


def preprocess_df(raw_dist_df, v, j, le_enc_v, le_enc_j, minor_genes_v=[], minor_genes_j=[]):
    dist_df = raw_dist_df[~raw_dist_df[v].isin(minor_genes_v)]
    dist_df = dist_df[~dist_df[j].isin(minor_genes_j)]
    
    dist_df[v] = le_enc_v.fit_transform(dist_df[v])
    dist_df[j] = le_enc_j.fit_transform(dist_df[j])
    
    skaler = MinMaxScaler()
    dist_df.iloc[:, 3:] = skaler.fit_transform(dist_df.iloc[:, 3:])
    
    return dist_df, skaler


def alphanumeric_sort(gene):
    return [int(i) if i.isdigit() else i.lower() for i in re.split(r'(\d+)', gene)]


def balance_majority(genes: pd.DataFrame, colu, min_count=0, max_count=1500):
    counts = genes[colu].value_counts()
    counts = counts.drop(counts[min_count>counts].index)
    resampled = pd.DataFrame()
    maj_clss = (counts[counts>max_count]).index
    left_genes = pd.DataFrame()
    mean_clss = counts[(counts<max_count) & (min_count<counts)].index#[i for i in genes[colu] if i not in min_classes]
    for cl in mean_clss:
        #print(cl)
        left_genes = pd.concat([left_genes, genes[genes[colu]==cl]])
    for maj_cl in maj_clss:        
        resampled = pd.concat([resampled, resample(genes[genes[colu] == maj_cl], replace=False, n_samples=max_count, random_state=42)])
    return pd.concat([left_genes, resampled])