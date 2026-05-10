import os
import sys
import csv
import numpy as np
import torch
from tqdm import tqdm
import librosa
import lightning.pytorch as pl

# Add root directory to path to import models and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clap_encoder import CLAP_Encoder
from utils import (
    load_ss_model,
    calculate_sdr,
    calculate_sisdr,
    parse_yaml,
)

class DCASEEvaluator:
    def __init__(
        self,
        sampling_rate=16000,
        eval_indexes='lass_synthetic_validation.csv',
        audio_dir='lass_validation',
    ) -> None:
        r"""DCASE T9 LASS evaluator.

        Returns:
            None
        """

        self.sampling_rate = sampling_rate

        if not os.path.exists(eval_indexes):
            raise FileNotFoundError(f"Evaluation index file not found: {eval_indexes}")

        with open(eval_indexes) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            eval_list = [row for row in csv_reader][1:]
        
        self.eval_list = eval_list
        self.audio_dir = audio_dir

    def __call__(
        self,
        pl_model: pl.LightningModule
    ) -> tuple:
        r"""Evaluate."""

        print(f'Evaluation on DCASE T9 synthetic validation set.')
        
        pl_model.eval()
        device = pl_model.device

        sisdrs_list = []
        sdris_list = []
        sdrs_list = []

        with torch.no_grad():
            for eval_data in tqdm(self.eval_list):
                # source, noise, snr, caption
                source_name, noise_name, snr, caption = eval_data
                snr = int(snr)

                source_path = os.path.join(self.audio_dir, f'{source_name}.wav')
                noise_path = os.path.join(self.audio_dir, f'{noise_name}.wav')

                if not os.path.exists(source_path) or not os.path.exists(noise_path):
                    continue

                source_audio, _ = librosa.load(source_path, sr=self.sampling_rate, mono=True)
                noise_audio, _ = librosa.load(noise_path, sr=self.sampling_rate, mono=True)

                # create audio mixture with a specific SNR level
                source_power = np.mean(source_audio ** 2)
                noise_power = np.mean(noise_audio ** 2)
                desired_noise_power = source_power / (10 ** (snr / 10))
                scaling_factor = np.sqrt(desired_noise_power / (noise_power + 1e-10))
                noise_audio = noise_audio * scaling_factor

                mixture = source_audio + noise_audio

                # declipping if need be
                max_value = np.max(np.abs(mixture))
                if max_value > 1:
                    source_audio *= 0.9 / max_value
                    mixture *= 0.9 / max_value

                sdr_no_sep = calculate_sdr(ref=source_audio, est=mixture)

                # Model expects (batch, channels, samples)
                # Query encoder expects text list
                conditions = pl_model.query_encoder.get_query_embed(
                    modality='text',
                    text=[caption],
                    device=device 
                )
                    
                input_dict = {
                    "mixture": torch.Tensor(mixture)[None, None, :].to(device),
                    "condition": conditions,
                }
                
                sep_segment = pl_model.ss_model(input_dict)["waveform"]
                # sep_segment: (batch_size=1, channels_num=1, segment_samples)

                sep_segment = sep_segment.squeeze().data.cpu().numpy()
                # sep_segment: (segment_samples,)
                
                # Match lengths if necessary (though they should be the same)
                min_len = min(len(source_audio), len(sep_segment))
                source_audio = source_audio[:min_len]
                sep_segment = sep_segment[:min_len]

                sdr = calculate_sdr(ref=source_audio, est=sep_segment)
                sdri = sdr - sdr_no_sep
                sisdr = calculate_sisdr(ref=source_audio, est=sep_segment)

                sisdrs_list.append(sisdr)
                sdris_list.append(sdri)
                sdrs_list.append(sdr)
        
        mean_sdri = np.mean(sdris_list) if sdris_list else 0
        mean_sisdr = np.mean(sisdrs_list) if sisdrs_list else 0
        mean_sdr = np.mean(sdrs_list) if sdrs_list else 0
        
        return mean_sisdr, mean_sdri, mean_sdr
    


def eval_script(checkpoint_path, eval_indexes, audio_dir, config_yaml='config/audiosep_base.yaml', device = "cuda"):
    configs = parse_yaml(config_yaml)

    # Load model
    query_encoder = CLAP_Encoder().eval()

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
        query_encoder=query_encoder
    ).to(device)

    print(f'-------  Start Evaluation  -------')

    evaluator = DCASEEvaluator(
        sampling_rate=16000,
        eval_indexes=eval_indexes,
        audio_dir=audio_dir,
    )

    # evaluation 
    SISDR, SDRi, SDR = evaluator(pl_model)
    msg = "SDR: {:.3f}, SDRi: {:.3f}, SISDR: {:.3f}".format(SDR, SDRi, SISDR)
    print(msg)

    print('-------------------------  Done  ---------------------------')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--eval_indexes', type=str, default='lass_synthetic_validation.csv')
    parser.add_argument('--audio_dir', type=str, default='lass_validation')
    parser.add_argument('--config_yaml', type=str, default='config/audiosep_base.yaml')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    eval_script(
        checkpoint_path=args.checkpoint_path,
        eval_indexes=args.eval_indexes,
        audio_dir=args.audio_dir,
        config_yaml=args.config_yaml,
        device=args.device
    )
