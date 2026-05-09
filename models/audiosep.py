from typing import Any, Callable, Dict
import random
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics

from models.clap_encoder import CLAP_Encoder

from huggingface_hub import PyTorchModelHubMixin


class AudioSep(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        ss_model: nn.Module = None,
        waveform_mixer = None,
        query_encoder: nn.Module = CLAP_Encoder().eval(),
        loss_function = None,
        optimizer_type: str = None,
        learning_rate: float = None,
        lr_lambda_func = None,
        use_text_ratio: float =1.0,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            ss_model: nn.Module
            anchor_segment_detector: nn.Module
            loss_function: function or object
            learning_rate: float
            lr_lambda: function
        """

        super().__init__()
        self.ss_model = ss_model
        self.waveform_mixer = waveform_mixer
        self.query_encoder = query_encoder
        self.query_encoder_type = self.query_encoder.encoder_type
        self.use_text_ratio = use_text_ratio
        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func
        
        # Validation Metric
        self.val_sdr = torchmetrics.audio.SignalDistortionRatio()


    def forward(self, x):
        pass

    def training_step(self, batch_data_dict, batch_idx):
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: e.g. 
                'audio_text': {
                    'text': ['a sound of dog', ...]
                    'waveform': (batch_size, 1, samples)
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        # [important] fix random seeds across devices
        random.seed(batch_idx)

        batch_audio_text_dict = batch_data_dict['audio_text']

        batch_text = batch_audio_text_dict['text']
        batch_audio = batch_audio_text_dict['waveform']
        device = batch_audio.device
        
        mixtures, segments = self.waveform_mixer(
            waveforms=batch_audio
        )

        # calculate text embed for audio-text data
        if self.query_encoder_type == 'CLAP':
            conditions = self.query_encoder.get_query_embed(
                modality='hybird',
                text=batch_text,
                audio=segments.squeeze(1),
                use_text_ratio=self.use_text_ratio,
            )

        input_dict = {
            'mixture': mixtures[:, None, :].squeeze(1),
            'condition': conditions,
        }

        target_dict = {
            'segment': segments.squeeze(1),
        }

        self.ss_model.train()
        sep_segment = self.ss_model(input_dict)['waveform']
        sep_segment = sep_segment.squeeze()
        # (batch_size, 1, segment_samples)

        output_dict = {
            'segment': sep_segment,
        }

        # Calculate loss.
        loss = self.loss_function(output_dict, target_dict)

        self.log_dict({"train_loss": loss})
        
        return loss

    def validation_step(self, batch_data_dict, batch_idx):
        r"""Forward a mini-batch validation data to model and calculate SDR.
        """
        batch_audio_text_dict = batch_data_dict['audio_text']

        batch_text = batch_audio_text_dict['text']
        batch_audio = batch_audio_text_dict['waveform'] # Ground truth
        
        # For validation, we use the pre-mixed 'mixture' if available, 
        # otherwise we fallback to the waveform_mixer (though val set should be pre-mixed)
        if 'mixture' in batch_audio_text_dict:
            mixtures = batch_audio_text_dict['mixture']
            segments = batch_audio # Ground truth is already in segments
        else:
            mixtures, segments = self.waveform_mixer(waveforms=batch_audio)

        device = mixtures.device
        
        # calculate text embed for audio-text data
        if self.query_encoder_type == 'CLAP':
            conditions = self.query_encoder.get_query_embed(
                modality='text',
                text=batch_text,
            )

        input_dict = {
            'mixture': mixtures[:, None, :].squeeze(1),
            'condition': conditions,
        }

        self.ss_model.eval()
        with torch.no_grad():
            sep_segment = self.ss_model(input_dict)['waveform']
            sep_segment = sep_segment.squeeze()

        # Calculate SDR
        # target: (batch_size, segment_samples), preds: (batch_size, segment_samples)
        # Move to CPU to avoid CUFFT_INTERNAL_ERROR which sometimes occurs on GPU
        device = sep_segment.device
        sep_segment_cpu = sep_segment.detach().cpu()
        segments_cpu = segments.squeeze().detach().cpu()

        # Move metric to CPU, calculate, and move back
        self.val_sdr.to('cpu')
        sdr_val = self.val_sdr(sep_segment_cpu, segments_cpu)
        self.val_sdr.to(device)

        self.log("val_sdr", sdr_val, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return sdr_val

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        r"""Configure optimizer.
        """

        if self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                params=self.ss_model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
                amsgrad=True,
            )
        else:
            raise NotImplementedError

        scheduler = LambdaLR(optimizer, self.lr_lambda_func)

        output_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

        return output_dict
    

def get_model_class(model_type):
    if model_type == 'ResUNet30':
        from models.resunet import ResUNet30
        return ResUNet30

    else:
        raise NotImplementedError
