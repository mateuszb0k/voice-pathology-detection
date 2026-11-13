import json

import numpy as np
import pandas as pd
import tqdm
from audio_processor import load_and_clean_audio, pad_or_truncate
from sklearn.preprocessing import StandardScaler

from src.CNN.feature_extractor import extract_mel_spectrograms


def process_split(csv_path,split_name,features_path):

    df = pd.read_csv(csv_path)
    features = pd.read_csv(features_path)
    df = pd.merge(df,features,how='inner',on=['FilePath','Status'])

    X = []
    y = []
    groups = []
    feats = []

    for _,row in tqdm.tqdm(df.iterrows()):
        path = f"../../data/wav/{row['FilePath']}"
        clean_audio = load_and_clean_audio(path)

        if len(clean_audio)==0:
            continue

        window = pad_or_truncate(clean_audio)
        mel_spectrogram= extract_mel_spectrograms(window)

        X.append(mel_spectrogram)
        y.append(row['Status'])
        groups.append(row['Id'])
        feats.append(row[4:].to_numpy(dtype=np.float32))

    X=np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    feats = np.array(feats)

    np.save(f"../../data/processed/{split_name}_features.npy",feats)
    np.save(f'../../data/processed/{split_name}_X.npy',X)
    np.save(f'../../data/processed/{split_name}_y.npy',y)
    np.save(f'../../data/processed/{split_name}_groups.npy',groups)
if __name__ == '__main__':
    scaler = StandardScaler()

    train_csv_path = '../../data/metadata/train.csv'
    val_csv_path = '../../data/metadata/val.csv'
    test_csv_path = '../../data/metadata/test.csv'
    features_csv_path = '../../data/metadata/features.csv'

    process_split(train_csv_path,'train',features_csv_path)
    process_split(val_csv_path,'val',features_csv_path)
    process_split(test_csv_path,'test',features_csv_path)

    train_X = np.load('../../data/processed/train_X.npy', allow_pickle=True)
    val_X = np.load('../../data/processed/val_X.npy', allow_pickle=True)
    test_X = np.load('../../data/processed/test_X.npy', allow_pickle=True)

    train_features = np.load('../../data/processed/train_features.npy', allow_pickle=True)
    val_features = np.load('../../data/processed/val_features.npy', allow_pickle=True)
    test_features = np.load('../../data/processed/test_features.npy', allow_pickle=True)

    #spectrogram normalization
    mean_train_X = np.mean(train_X)
    std_train_X = np.std(train_X)
    d = {"mean_train_X":str(mean_train_X),"std_train_X":str(std_train_X)}
    with open('../../data/metadata/normalization_params.json', 'w') as f:
        json.dump(d, f)
    train_X = (train_X - mean_train_X) / std_train_X
    val_X = (val_X - mean_train_X) / std_train_X
    test_X = (test_X - mean_train_X) / std_train_X
    np.save('../../data/processed/train_X.npy', train_X)
    np.save('../../data/processed/val_X.npy', val_X)
    np.save('../../data/processed/test_X.npy', test_X)

    #features normalization
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
    np.save('../../data/processed/train_features.npy', train_features)
    np.save('../../data/processed/val_features.npy', val_features)
    np.save('../../data/processed/test_features.npy', test_features)

    params = {"mean_train_features" : scaler.mean_.tolist() ,"std_train_features" : scaler.scale_.tolist()}

    with open('../../data/metadata/features_normalization_params.json', 'w') as f:
        json.dump(params, f)