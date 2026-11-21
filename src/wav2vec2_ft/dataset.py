import librosa as lr
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from audio_utils import pad_right_or_center_crop, load_and_clean_audio


class SVDWaveFormDataset(Dataset):

    def __init__(self, path):
        self.manifest = pd.read_csv(path)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        x = load_and_clean_audio('../../data/wav/' + self.manifest['FilePath'][idx])
        x, valid_length= pad_right_or_center_crop(x, sr=16000)
        x = torch.from_numpy(x).type(torch.float32)
        y = self.manifest['Status'][idx]

        return x, y, valid_length


def get_dataloaders(batch_size=32):
    train_dataset = SVDWaveFormDataset('../../data/processed/train_manifest.csv')
    val_dataset = SVDWaveFormDataset('../../data/processed/val_manifest.csv')
    test_dataset = SVDWaveFormDataset('../../data/processed/test_manifest.csv')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
