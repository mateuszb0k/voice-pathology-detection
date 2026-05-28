import librosa as lr
import numpy as np
from audio_processor import load_and_clean_audio,extract_windows
def extract_mel_spectrograms(windows,sr: int = 16000,n_fft: int =1024, hop_length:int =256,n_mels:int=128):
    spectrograms = []
    for window in windows:
        spectrogram = lr.feature.melspectrogram(y=window,sr = sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        spectrogram_db = lr.power_to_db(spectrogram, ref=np.max)
        spectrograms.append(spectrogram_db)
    spectrograms = np.array(spectrograms)
    return spectrograms
if __name__ == "__main__":
    PATH = "../data/raw/2_2-u_n.wav"
    clean_audio = load_and_clean_audio(PATH)
    windows = extract_windows(clean_audio)
    spectrograms = extract_mel_spectrograms(windows)
    print(spectrograms.shape)