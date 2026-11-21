import numpy as np
import librosa as lr


def load_and_clean_audio(file_path:str, target_sr: int = 16000, top_db: int = 30):
    y,sr = lr.load(file_path, sr=target_sr)

    if y is None or len(y) == 0:
        return np.array([], dtype=np.float32)

    y= y-np.mean(y)
    trimmed,_ = lr.effects.trim(y, top_db=top_db)

    if trimmed is None or len(trimmed) == 0:
        return np.array([], dtype=np.float32)

    max_amp = np.max(np.abs(trimmed))

    if max_amp!=0 and len(trimmed)>0:
        normalized = trimmed/max_amp
        return normalized
    else:
        return trimmed

def pad_right_or_center_crop(signal: np.ndarray, sr: int = 16000, duration: int = 3):
    target_length = sr * duration
    signal_length = len(signal)

    valid_length = min(target_length, signal_length)

    if signal_length <= 0:
        raise ValueError("Signal is empty")

    if signal_length > target_length:
        start = (signal_length - target_length) // 2
        cropped_signal = signal[start:start + target_length]

        return cropped_signal.astype(np.float32), valid_length

    else:
        diff = target_length - signal_length

        padded_signal = np.pad(signal, (0, diff), mode='constant')

        return padded_signal.astype(np.float32), valid_length

