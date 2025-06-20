import pickle 
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


import pickle 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score



def create_df_embs(model, resampled_df, device='cpu'):
    hid_sts = []
    model.to(device)

    for seq in tqdm(resampled_df.index):
        
        seq = resampled_df.loc[seq][0]
        en_dict = model.tokenizer.encode_plus(seq, 
                                              add_special_tokens=True,
                                              max_length=27,
                                              padding='max_length',
                                              return_attention_mask=True,
                                              return_tensors='pt')
        
        input_ids_test, att_mask_test = en_dict['input_ids'], en_dict['attention_mask']

        input_ids_test = input_ids_test.to(device)
        att_mask_test= att_mask_test.to(device)

        with torch.no_grad():
            outputs = model.model(input_ids_test, att_mask_test, output_hidden_states=True)


        emb = torch.mean(torch.mean(torch.mean(torch.stack(list(outputs.hidden_states)), dim=2), dim=1), dim =0)
        hid_sts.append(emb.tolist())
        
    # df_emb = pd.concat([resampled_df, pd.DataFrame(hid_sts)], axis=1)

    return pd.DataFrame(hid_sts)

def train_clf(X_train, X_test, y_train, y_test, gene='j', param_grid=None, save=True, path=None):
    # df_emb.drop(columns = ['cdr3aa'], inplace=True)
    # X_train, X_test, y_train, y_test = train_test_split(df_emb.drop([gene], axis=1), df_emb[gene], test_size=0.20, random_state=42)
    
    if not param_grid:
        param_grid = {'n_neighbors': range(5, 41, 5), 'weights' : ['uniform', 'distance']}
    
    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=1, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best CV score: {:.3f}, best CV k: {}".format(grid_search.best_score_, grid_search.best_estimator_)) 
    
    test_predictions = grid_search.best_estimator_.predict(X_test)
    print("Resulting test score: {:.3f}".format(f1_score(test_predictions, y_test, average='macro')))
    
    knn_best = grid_search.best_estimator_
    
    if save:
        path = f'knn_{gene}_model.pkl' if not path else path
        with open(path, 'wb') as file:
            pickle.dump(knn_best, file)
    
    return knn_best  

def get_nearest_neighbours(path_to_model, model, seq_inp, X_train, y_train, n_neighb=None, print_info=False):
    device = 'cpu'
    with open(path_to_model, 'rb') as file:
        knn_model = pickle.load(file)
        
    model.to(device)
    en_dict = model.tokenizer.encode_plus(seq_inp, add_special_tokens = True, 
                                      max_length = 25, pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')
    input_ids_test, att_mask_test = en_dict['input_ids'], en_dict['attention_mask']

    input_ids_test = input_ids_test.to(device)
    att_mask_test= att_mask_test.to(device)

    with torch.no_grad():
        outputs = model.model(input_ids_test, att_mask_test, output_hidden_states=True)
        
    emb = torch.mean(torch.mean(torch.mean(torch.stack(list(outputs.hidden_states)), dim=2), dim=1), dim =0)
    seq = emb.tolist()
    
    
    distances, indices = knn_model.kneighbors([seq])
    if n_neighb:
        nearest_neighbors = X_train.iloc[indices[0]].iloc[:n_neighb,]
        nearest_labels = y_train.iloc[indices[0]].iloc[:n_neighb,]
    else:
        nearest_neighbors = X_train.iloc[indices[0]]
        nearest_labels = y_train.iloc[indices[0]]
        
    if print_info:
        print("Query Point:", seq_inp)
        print("Nearest Neighbors:", list(nearest_neighbors.index))
        print("Nearest Labels:", nearest_labels.values)
        print("Distances to Nearest Neighbors:", distances)
        
    indexes = nearest_neighbors.index
    return indexes

    