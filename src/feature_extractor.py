import librosa as lr
import numpy as np
from audio_processor import load_and_clean_audio,pad_or_truncate
def extract_mel_spectrograms(window,sr: int = 16000,n_fft: int =1024, hop_length:int =256,n_mels:int=128):
    spectrogram = lr.feature.melspectrogram(y=window,sr = sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spectrogram_db = lr.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db
if __name__ == "__main__":
    PATH = "../data/raw/2_2-u_n.wav"
    clean_audio = load_and_clean_audio(PATH)
    window = pad_or_truncate(clean_audio)
    spectrograms = extract_mel_spectrograms(window)
    print(spectrograms.shape)