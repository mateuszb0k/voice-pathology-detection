import time

import numpy as np
import pandas as pd
import torch
from torchaudio.pipelines import WAV2VEC2_BASE as bundle
from tqdm import tqdm

from src.data_prep.audio_processor import load_and_clean_audio, pad_or_truncate


def main(split='train'):
    if split == 'train':
        data = pd.read_csv("../../data/metadata/train.csv")
    elif split == 'val':
        data = pd.read_csv("../../data/metadata/val.csv")
    else:
        raise ValueError('Split must be "train" or "val"')
    features = pd.read_csv("../../data/metadata/features.csv")

    data = data.merge(features, how='inner', on=['FilePath', 'Status'])[['FilePath', 'Status', 'Id']]

    print(data.shape)
    print(data.head(3))

    model = bundle.get_model()
    model.requires_grad_(False)
    model.eval()

    result = []

    default_path = "../../data/wav/"

    t1 = time.perf_counter()

    with torch.no_grad():
        for row in tqdm(data.iterrows(), total=len(data)):
            path = row[1]['FilePath']
            wav = load_and_clean_audio(default_path + path)

            if wav.size > 0:
                wav = pad_or_truncate(wav)
            else:
                raise ValueError(f'File path {path} contains an empty file.')

            wav = torch.from_numpy(wav)
            wav = wav.unsqueeze(0)

            out, _ = model(wav)
            out = out.mean(dim=1)

            result.append(out)
    t2 = time.perf_counter()
    diff = t2 - t1

    print(diff)

    result = torch.cat(result, dim=0)
    print(result.shape, result.dtype)

    if not torch.isfinite(result).all().item():
        raise ValueError('Result is not finite')

    result = result.cpu().numpy()
    data.to_csv(f"../../data/processed/{split}_manifest.csv", index=False)
    np.save(f"../../data/processed/{split}_embeddings.npy", result)

    d = np.load(f"../../data/processed/{split}_embeddings.npy")
    m = pd.read_csv(f"../../data/processed/{split}_manifest.csv")

    print(d.shape)
    print(m.shape)
    print(m.equals(data.reset_index(drop=True)))


if __name__ == "__main__":
    main(split='val')
