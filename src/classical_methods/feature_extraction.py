import opensmile
import pathlib
import pandas as pd
import librosa as lr
import numpy as np
from tqdm import tqdm

SAMPLING_RATE = 16_000
TOP_DB = 30
SMILE = opensmile.Smile(
    feature_set = opensmile.FeatureSet.eGeMAPSv02,
    feature_level = opensmile.FeatureLevel.Functionals
)

def load_and_clean_audio(file_path: str, target_sr: int, top_db: int):
    y,sr = lr.load(file_path,sr = target_sr)
    if y is None or len(y) == 0:
        return np.array([], dtype=np.float32)
    y = y-np.mean(y)
    trimmed,_ = lr.effects.trim(y, top_db=top_db)
    if trimmed is None:
        return np.array([], dtype=np.float32)
    return trimmed


def extract_features(file_path: str, sr: int, top_db: int):
    signal = load_and_clean_audio(file_path, sr, top_db)
    if len(signal) < sr * 0.25:
        return np.array([], dtype=np.float32)
    else:
        features = SMILE.process_signal(
            signal,
            sr
        )
        return features

if __name__ == "__main__":
    METADATA_PATH = pathlib.Path("../../data/metadata")
    AUDIO_DIR = pathlib.Path("../../data/wav")
    TEST_SPLIT = pd.read_csv(METADATA_PATH / 'test.csv')
    TRAIN_SPLIT = pd.read_csv(METADATA_PATH / 'train.csv')
    VAL_SPLIT = pd.read_csv(METADATA_PATH / 'val.csv')
    combined_splits = pd.concat([TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT] , ignore_index=True)
    feature_dfs = []
    skipped = []
    for row in tqdm(combined_splits.itertuples(index = False)):
        file_path = AUDIO_DIR / row.FilePath
        features = extract_features(str(file_path), SAMPLING_RATE, TOP_DB)
        if len(features) == 0:
            print("Skipping file {}".format(row.FilePath))
            skipped.append({"FilePath": row.FilePath})
            continue
        else:
            features['FilePath'] = row.FilePath
            features['Status'] = row.Status
            feature_dfs.append(features)
    skipped = pd.DataFrame.from_records(skipped)
    print(skipped.head(10))
    skipped.to_csv("../../data/metadata/skipped_features.csv", index=False)
    frame = pd.concat(feature_dfs, ignore_index=True)
    frame.to_csv(METADATA_PATH / 'features.csv', index=False)