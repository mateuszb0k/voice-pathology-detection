import pandas as pd
import numpy as np
import tqdm
from audio_processor import load_and_clean_audio,extract_windows
from feature_extractor import extract_mel_spectrograms
import json
def process_split(csv_path,split_name):
    df = pd.read_csv(csv_path)
    X = []
    y = []
    groups = []
    for _,row in tqdm.tqdm(df.iterrows()):
        path = f"../data/raw/{row['FilePath']}"
        clean_audio = load_and_clean_audio(path)
        if len(clean_audio)==0:
            continue
        windows = extract_windows(clean_audio)
        mel_spectrograms = extract_mel_spectrograms(windows)
        for mel_spectrogram in mel_spectrograms:
            X.append(mel_spectrogram)
            y.append(row['Status'])
            groups.append(row['Id'])
    X=np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    np.save(f'../data/processed/{split_name}_X.npy',X)
    np.save(f'../data/processed/{split_name}_y.npy',y)
    np.save(f'../data/processed/{split_name}_groups.npy',groups)
if __name__ == '__main__':
    # train_csv_path = '../data/metadata/train.csv'
    # val_csv_path = '../data/metadata/val.csv'
    # test_csv_path = '../data/metadata/test.csv'
    # process_split(train_csv_path,'train')
    # process_split(val_csv_path,'val')
    # process_split(test_csv_path,'test')
    train_X = np.load('../data/processed/train_X.npy',allow_pickle=True)
    val_X = np.load('../data/processed/val_X.npy',allow_pickle=True)
    test_X = np.load('../data/processed/test_X.npy',allow_pickle=True)
    mean_train_X = np.mean(train_X)
    std_train_X = np.std(train_X)
    d = {"mean_train_X":str(mean_train_X),"std_train_X":str(std_train_X)}
    with open(f'../data/metadata/normalization_params.json','w') as f:
        json.dump(d, f)
    train_X = (train_X - mean_train_X) / std_train_X
    val_X = (val_X - mean_train_X) / std_train_X
    test_X = (test_X - mean_train_X) / std_train_X
    np.save(f'../data/processed/train_X.npy',train_X)
    np.save(f'../data/processed/val_X.npy',val_X)
    np.save(f'../data/processed/test_X.npy',test_X)
