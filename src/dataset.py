import torch
from torch.utils.data import DataLoader,Dataset
from torchaudio.transforms import FrequencyMasking,TimeMasking
import numpy as np
class VoicePathologyDataset(Dataset):
    def __init__(self,x_path,y_path,is_train):
        self.X = np.load(x_path,allow_pickle=True)
        self.y = np.load(y_path,allow_pickle=True)
        self.is_train = is_train
        self.freq_masking = FrequencyMasking(freq_mask_param = 15)
        self.time_masking = TimeMasking(time_mask_param = 15)
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        x = self.X[idx]
        y = self.y[idx]
        x = np.expand_dims(x,axis=0)
        x = torch.tensor(x,dtype=torch.float32)
        y = torch.tensor(y,dtype=torch.long)
        if self.is_train:
            x = self.freq_masking(x)
            x = self.time_masking(x)
        return x,y
def get_data_loaders():
    train_dataset = VoicePathologyDataset(x_path = '../data/processed/train_X.npy',y_path = '../data/processed/train_y.npy',is_train = True)
    val_dataset = VoicePathologyDataset(x_path = '../data/processed/val_X.npy',y_path = '../data/processed/val_y.npy',is_train = False)
    test_dataset = VoicePathologyDataset(x_path = '../data/processed/test_X.npy',y_path = '../data/processed/test_y.npy',is_train = False)
    train_data_loader = DataLoader(train_dataset,batch_size = 32,shuffle = True)
    val_data_loader = DataLoader(val_dataset,batch_size = 32,shuffle = False)
    test_data_loader = DataLoader(test_dataset,batch_size = 32,shuffle = False)
    return train_data_loader,val_data_loader,test_data_loader