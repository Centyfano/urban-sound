from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import torch

class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annotations_file = os.path.abspath(annotations_file)
        self.annotations = pd.read_csv(self.annotations_file)
        self.audio_dir = os.path.abspath(audio_dir)
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sr = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self._truncate_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _truncate_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        sig_len = signal.shape[1]
        if sig_len < self.num_samples:
            num_miss = self.num_samples - sig_len
            last_dim_padding = (0, num_miss)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        """Resample all the data so that all of them has the same sample rate"""
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        """
            Mixes down a multichannel signal tensor to mono if necessary.

            Args:
                signal (tensor): The signal tensor to be mixed down. Has shape `(n_channels, n_samples)`.

            Returns:
                mixed_signal (tensor): The mixed down signal tensor. Will have shape `(1, n_samples)` if the input 
                signal was stereo, or (n_channels, n_samples) if the input signal was already mono.

            This method checks if the input signal is stereo or mono. If the signal is stereo (i.e., has two channels), it is mixed down to mono by computing the mean of the two channels. If the signal is already mono (i.e., has only one channel), it is returned unchanged.

            The input signal tensor must have shape (n_channels, n_samples), where  n_channels is the number of channels. The output mixed down signal tensor will have shape (batch_size, n_samples) if the input signal was stereo, or (batch_size, n_samples, n_channels) if the input signal was already mono.
            """
        if signal.shape[0]  > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]



if __name__ == "__main__":

    d_path = "data/UrbanSound8K"
    ANNOTATIONS_FILE = f"{d_path}/metadata/UrbanSound8K.csv"
    AUDIO_DIR = f"{d_path}/audio"
    SR = 22050
    NUM_SAMPLES = 22050

    
    device = "cuda" if torch.cuda.is_available() else "cpu"


    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=1024,
        hop_length=512, 
        n_mels=64
    )

    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE, 
        AUDIO_DIR, 
        transformation=mel_spectogram, 
        target_sample_rate=SR, 
        num_samples = NUM_SAMPLES,
        device=device
    )
    
    print(f"There are {len(usd)} samples in the dataset")
    signal, label = usd[0]

    a =1

