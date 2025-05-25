import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import sys
import datetime
import math
import random
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

if torch.cuda.is_available():
    device = torch.device("cuda")
    # print('There are %d GPU(s) available.' % torch.cuda.device_count())
    # print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    # print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/gan/covid_diploma/{current_time}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

import torch.nn.functional as F


parent_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.append(parent_dir)
sys.path.append(os.path.abspath(os.path.join(parent_dir, '..')))

# from utilities import *

sys.path.append(os.path.join('/home/akabalina', 'OLGA/olga'))
# sys.path



import olga.load_model as load_model
import olga.generation_probability as pgen
params_file_name = '/home/akabalina/OLGA/olga/default_models/human_T_beta/model_params.txt'
marginals_file_name = '/home/akabalina/OLGA/olga/default_models/human_T_beta/model_marginals.txt'
V_anchor_pos_file ='/home/akabalina/OLGA/olga/default_models/human_T_beta/V_gene_CDR3_anchors.csv'
J_anchor_pos_file = '/home/akabalina/OLGA/olga/default_models/human_T_beta/J_gene_CDR3_anchors.csv'
genomic_data = load_model.GenomicDataVDJ()
genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)

generative_model = load_model.GenerativeModelVDJ()
generative_model.load_and_process_igor_model(marginals_file_name)

pgen_model = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)


def pgen_log(pgen_val):
    if pgen_val == 0:
        pgen_val += 1e-20
    return round(math.log10(pgen_val), 5)





seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)

kidera = pd.DataFrame.from_records(list(map(lambda x: list(map(float, x.split(','))), "-1.56,-1.67,-0.97,-0.27,-0.93,-0.78,-0.2,-0.08,0.21,-0.48;0.22,1.27,1.37,1.87,-1.7,0.46,0.92,-0.39,0.23,0.93;1.14,-0.07,-0.12,0.81,0.18,0.37,-0.09,1.23,1.1,-1.73;0.58,-0.22,-1.58,0.81,-0.92,0.15,-1.52,0.47,0.76,0.7;0.12,-0.89,0.45,-1.05,-0.71,2.41,1.52,-0.69,1.13,1.1;-0.47,0.24,0.07,1.1,1.1,0.59,0.84,-0.71,-0.03,-2.33;-1.45,0.19,-1.61,1.17,-1.31,0.4,0.04,0.38,-0.35,-0.12;1.46,-1.96,-0.23,-0.16,0.1,-0.11,1.32,2.36,-1.66,0.46;-0.41,0.52,-0.28,0.28,1.61,1.01,-1.85,0.47,1.13,1.63;-0.73,-0.16,1.79,-0.77,-0.54,0.03,-0.83,0.51,0.66,-1.78;-1.04,0,-0.24,-1.1,-0.55,-2.05,0.96,-0.76,0.45,0.93;-0.34,0.82,-0.23,1.7,1.54,-1.62,1.15,-0.08,-0.48,0.6;-1.4,0.18,-0.42,-0.73,2,1.52,0.26,0.11,-1.27,0.27;-0.21,0.98,-0.36,-1.43,0.22,-0.81,0.67,1.1,1.71,-0.44;2.06,-0.33,-1.15,-0.75,0.88,-0.45,0.3,-2.3,0.74,-0.28;0.81,-1.08,0.16,0.42,-0.21,-0.43,-1.89,-1.15,-0.97,-0.23;0.26,-0.7,1.21,0.63,-0.1,0.21,0.24,-1.15,-0.56,0.19;0.3,2.1,-0.72,-1.57,-1.16,0.57,-0.48,-0.4,-2.3,-0.6;1.38,1.48,0.8,-0.56,0,-0.68,-0.31,1.03,-0.05,0.53;-0.74,-0.71,2.04,-0.4,0.5,-0.81,-1.07,0.06,-0.46,0.65".split(";"))), index=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"], columns=list(map(lambda x: "f"+str(x), range(1,11))))
# https://github.com/vadimnazarov/kidera-atchley?tab=readme-ov-file

aa_inds = pd.DataFrame(columns=kidera.columns, index = range(0,20))
for i in range(10):
    curr_kidera = kidera.columns[i]
    aa_inds[curr_kidera] = kidera[curr_kidera].sort_values().index.values
    # print(kidera[curr_kidera].sort_values().index.values)



def encode_seq(seq):
    mtr = np.zeros((10, len(seq), 20))
    for factor in range(10):
        for aa_i in range(len(seq)):
            mtr[factor, aa_i, list(aa_inds[f'f{factor+1}']).index(seq[aa_i])] = 1
            # continue
            # print(list(aa_inds[f'f{factor+1}']).index(seq[aa_i]))#==)#].index)
        
    return mtr#.shape



b_blos = pd.read_csv('covid_data_seqs_pgen.csv') #pd.read_pickle('olga_enc.pkl')
b_blos['seq_enc'] = b_blos['cdr3aa'].apply(encode_seq)



train_ds = torch.tensor(b_blos['seq_enc'].tolist(), dtype=torch.float32)
train_ps = torch.tensor(b_blos['pgen_log'].tolist(), dtype=torch.float32)



# Create PyTorch DataLoader object to produce batches
batch_size = 32
train_dl = DataLoader(train_ds, batch_size, shuffle=False, pin_memory=True)
train_pl = DataLoader(train_ps, batch_size, shuffle=False, pin_memory=True)

# Utils functions for GPU usage of neural networks
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class SoftExponential(nn.Module):
    def __init__(self, alpha=-0.5, beta=1000.0):
        super(SoftExponential, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        if self.alpha.item() < 0:
            return -1.0 * self.beta / self.alpha * (torch.exp(-self.alpha * x) - 1)
        elif self.alpha.item() == 0:
            return x * self.beta
        else:
            return 1.0 * self.beta / self.alpha * (torch.exp(self.alpha * x) - 1)

latent_size = 32

from torch.nn.utils import spectral_norm
# import torchvision.transforms as transforms


discriminator = nn.Sequential(
    spectral_norm(nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1, bias=False)),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.2, inplace=True),

    spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False)),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),

    spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    spectral_norm(nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False)),
    
    nn.Flatten(),
    nn.Linear(15, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    SoftExponential())



generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.Dropout(0.3),

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.Dropout(0.3),

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.Dropout(0.3),

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 10, kernel_size=(4, 5), stride=(1, 1), padding=(10, 8), bias=False),
    nn.Tanh()
)


device = get_default_device()

discriminator = to_device(discriminator, device)
generator = to_device(generator, device)


checkpoint_path = 'logs/gan/20250522-040142/checkpoints/epoch_19.pth'
checkpoint = torch.load(checkpoint_path)
opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
generator.load_state_dict(checkpoint['generator'])
discriminator.load_state_dict(checkpoint['discriminator'])
opt_g.load_state_dict(checkpoint['opt_g'])
opt_d.load_state_dict(checkpoint['opt_d'])



def train_discriminator(real_images, real_targets, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    real_preds = discriminator(real_images)
    real_targets = real_targets.to(device)
    
    # real_loss = F.binary_cross_entropy(real_preds, real_targets)
    mse = nn.MSELoss()
    real_loss = mse(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    # latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    latent = torch.rand(batch_size, latent_size, 1, 1, device=device) * 2 - 1

    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.empty(fake_images.size(0), 1, device=device).uniform_(-20, -18)
    fake_preds = discriminator(fake_images)
    # fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    mse = nn.MSELoss()

    fake_loss = mse(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score

def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate fake images
    # latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    latent = torch.rand(batch_size, latent_size, 1, 1, device=device) * 2 - 1

    fake_images = generator(latent)
    
    # Try to fool the discriminator
    preds = discriminator(fake_images)
    # targets = torch.ones(batch_size, 1, device=device)
    targets = torch.empty(batch_size, 1, device=device).uniform_(-18, -11)
    mse = nn.MSELoss()
    loss = mse(preds, targets)#F.binary_cross_entropy(preds, targets)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()

fixed_latent = torch.rand(batch_size, latent_size, 1, 1, device=device) * 2 - 1



def decode_seq(pred, aa_inds=aa_inds):
    pred = pred.argmax(dim=2)
    pred_seqs = []
    
    for i in range(pred.shape[0]):
        curr_code = aa_inds[f'f{i+1}']
        curr_seq = pred[i].tolist()
        # print('')
        dec_seq = []
        for j in range(15):
            # print(curr_seq[j])
            # print('')

            dec_seq.append(curr_code.loc[curr_seq[j]])
        dec_seq = ''.join(dec_seq)
        pgen = pgen_model.compute_aa_CDR3_pgen(dec_seq)
        pred_seqs.append((''.join(dec_seq), pgen_log(pgen)))
    
    pred_seqs.sort(key = lambda x: x[1], reverse=True)
    # decoded_seq = ''.join(reverse_aa[pred])
    # pgen = pgen_model.compute_aa_CDR3_pgen(decoded_seq)
    

    return pred_seqs[0]#decoded_seq, pgen




def histo_samples(sample_pgenes, show=True):

    if show:
        # plt.figure(figsize=(4, 3))
        sns.displot(sample_pgenes, kde=True, height=3, aspect=1.5)
        # plt.plot(sample_pgenes, '-')
        plt.xlabel('OLGA p_gen')
        plt.show()


def calculate_diversity(sequences):
    # sequences = list(map(lambda x: x[0], sequences))

    unique_seqs = set(sequences)
        
    diversity = len(unique_seqs) / len(sequences)
    return diversity

def fit(epochs, lr, start_idx=1, opt_d=None, opt_g=None):
    torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    checkpoin_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoin_dir, exist_ok=True)

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    all_pgen = []
    diversity = []
    
    if opt_d is None:
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    if opt_g is None:
        opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for real_images, real_targ in tqdm(zip(train_dl, train_pl)):
            loss_d, real_score, fake_score = train_discriminator(real_images, real_targ, opt_d)
            loss_g = train_generator(opt_g)
        
        
        latent = torch.rand(batch_size, latent_size, 1, 1, device=device) * 2 - 1
        samples = generator(latent)
            
        seqs_decoded = []
        pgenes = []
        for seq in samples:
            seq_curr, pgen = decode_seq(seq)
            seqs_decoded.append(seq_curr)
            pgenes.append(pgen)
            
        divers = calculate_diversity(seqs_decoded)
        
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        diversity.append(divers)
        all_pgen.append(pgenes)

        
        writer.add_scalar('Loss/Generator', loss_g, epoch)
        writer.add_scalar('Loss/Discriminator', loss_d, epoch)
        writer.add_scalar('Score/Real', real_score, epoch)
        writer.add_scalar('Score/Fake', fake_score, epoch)
        writer.add_scalar('Diversity', divers, epoch)
        
        writer.add_text('Generated_Sequences', ', '.join(seqs_decoded[:5]), epoch)
        
        writer.add_histogram('pgen_distribution', torch.tensor(pgenes), epoch)

        print(f"Epoch [{epoch+1}/{epochs}], loss_g: {loss_g:.4f}, loss_d: {loss_d:.4f}, "
              f"real_score: {real_score:.4f}, fake_score: {fake_score:.4f}, Diversity: {divers}")
        
        if (divers > 0.3) and (max(pgenes) > -10):     
            torch.save({'epoch': epoch,
                        'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'opt_g': opt_g.state_dict(),
                        'opt_d': opt_d.state_dict(),}, f'{checkpoin_dir}/epoch_{epoch+1}.pth')
    
    writer.close()
    
    history = {
        'losses_g': losses_g,
        'losses_d': losses_d,
        'real_scores': real_scores,
        'fake_scores': fake_scores, 
        'diversity': diversity, 
        'all_pgen': all_pgen, 
        'opt_d': opt_d, 
        'opt_g': opt_g}
    
    return history

lr = 0.00005
epochs = 100



train_dl = DeviceDataLoader(train_dl, device)
train_pl = DeviceDataLoader(train_pl, device)

history = fit(epochs, lr,  opt_d=opt_d, opt_g=opt_g)
import pickle
with open(log_dir + '/history.pkl', 'wb') as f:
    pickle.dump(history, f)
    
pgen_res_ = np.array(history['all_pgen'])
plt.figure(figsize=(12, 15))
sns.heatmap(pgen_res_, cmap="Greens", vmin=-20, vmax=-6, annot=False)
results_path = log_dir + '/results_big.png'
plt.savefig(results_path)