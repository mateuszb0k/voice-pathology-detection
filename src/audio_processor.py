
import librosa as lr
import numpy as np
import pandas as pd
def load_and_clean_audio(file_path:str, target_sr: int = 16000, top_db: int = 30):
    y,sr = lr.load(file_path, sr=target_sr)
    y= y-np.mean(y)
    trimmed,_ = lr.effects.trim(y, top_db=top_db)
    max_amp = np.max(np.abs(trimmed))
    if max_amp!=0 and len(trimmed)>0:
        normalized = trimmed/max_amp
        return normalized
    else:
        return trimmed
