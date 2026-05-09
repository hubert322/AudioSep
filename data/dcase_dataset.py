import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

class DCASEValidationDataset(Dataset):
    """
    Dataset for DCASE 2024 Task 9 Validation.
    Loads pre-mixed mixtures and ground truth targets.
    """
    def __init__(self, csv_path, audio_root, sampling_rate=32000, max_clip_len=5):
        self.df = pd.read_csv(csv_path, skipinitialspace=True)
        self.audio_root = audio_root
        self.sampling_rate = sampling_rate
        self.max_length = max_clip_len * sampling_rate

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or crop to max_length
        if waveform.size(1) > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            temp_wav = torch.zeros(1, self.max_length)
            temp_wav[:, :waveform.size(1)] = waveform
            waveform = temp_wav
            
        return waveform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        # Paths are build from 'source' and 'noise' columns + .wav
        source_path = os.path.join(self.audio_root, f"{row['source']}.wav")
        noise_path = os.path.join(self.audio_root, f"{row['noise']}.wav")
        snr = row['snr']
        caption = row['caption']
        
        # Load audio
        source = self._load_audio(source_path)
        noise = self._load_audio(noise_path)

        # --- Official DCASE Mixing Logic ---
        import torch.nn.functional as F
        import numpy as np

        # 1. Calculate power
        source_power = torch.mean(source ** 2)
        noise_power = torch.mean(noise ** 2)

        # 2. Calculate desired noise power based on SNR
        # SNR = 10 * log10(source_power / noise_power)
        desired_noise_power = source_power / (10 ** (snr / 10.0))

        # 3. Scale noise
        scaling_factor = torch.sqrt(desired_noise_power / (noise_power + 1e-10))
        noise = noise * scaling_factor

        # 4. Create mixture
        mixture = source + noise

        # 5. Safety Declipping (0.9 margin)
        max_val = torch.max(torch.abs(mixture))
        if max_val > 1.0:
            source = source * (0.9 / max_val)
            mixture = mixture * (0.9 / max_val)
        # -----------------------------------

        return {
            'text': caption,
            'mixture': mixture,
            'waveform': source, # The target we want to extract
            'modality': 'audio_text'
        }
