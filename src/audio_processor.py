
import librosa as lr
import numpy as np
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
def extract_windows(signal, window_size: int = 16000, hop_length: int = 8000) -> np.ndarray:
    windows = []
    signal_length = len(signal)
    if window_size > signal_length:
        times = np.ceil(window_size/signal_length)
        signal = np.tile(signal, (int(times)))
        signal = signal[:window_size]
        windows.append(signal)
        windows = np.array(windows)
        return windows
    else:
        for i in range(0,signal_length,hop_length):
            frame = signal[i:i+window_size]
            frame_length = len(frame)
            if frame_length<window_size:
                times = np.ceil(window_size/frame_length)
                frame = np.tile(frame, (int(times)))
                frame = frame[:window_size]
            windows.append(frame)
    windows = np.array(windows)
    return windows
if __name__ == "__main__":
    PATH = "../data/raw/2_2-u_n.wav"
    clean_audio = load_and_clean_audio(PATH)
    print(clean_audio.shape)
    frames = extract_windows(clean_audio)
    print(frames.shape)