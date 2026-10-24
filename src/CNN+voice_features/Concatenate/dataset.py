import torch
from torch.utils.data import DataLoader,Dataset
from torchaudio.transforms import FrequencyMasking,TimeMasking
import numpy as np
class ConcatenateDataset(Dataset):
    def __init__(self,spec_path,feats_path,y_path,is_train):
        self.X = np.load(spec_path,allow_pickle=True)
        self.Y = np.load(y_path,allow_pickle=True)
        self.feats = np.load(feats_path,allow_pickle=True)

        self.is_train = is_train

        self.freq_masking = FrequencyMasking(freq_mask_param = 15)
        self.time_masking = TimeMasking(time_mask_param = 20)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.Y[idx]
        feat = self.feats[idx]

        x = np.expand_dims(x,axis=0)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        feats = torch.tensor(feat, dtype=torch.float32)

        if self.is_train:
            x = self.time_masking(x)
            x = self.freq_masking(x)
            pass

        return x,feats,y
def get_dataloaders():

    train_dataset = ConcatenateDataset(spec_path = '../../../data/processed/train_X.npy',
                                       y_path = '../../../data/processed/train_y.npy',
                                       feats_path='../../../data/processed/train_features.npy' ,is_train = True)
    val_dataset = ConcatenateDataset(spec_path='../../../data/processed/test_X.npy',
                                       y_path='../../../data/processed/test_y.npy',
                                       feats_path='../../../data/processed/test_features.npy', is_train=False)
    test_dataset = ConcatenateDataset(spec_path='../../../data/processed/test_X.npy',
                                      y_path='../../../data/processed/test_y.npy',
                                      feats_path='../../../data/processed/test_features.npy', is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader