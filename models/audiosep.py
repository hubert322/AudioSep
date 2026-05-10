import logging
from typing import Any, Callable, Dict
import random
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics

from models.clap_encoder import CLAP_Encoder
from models.metrics import calculate_sdr_torch, calculate_sisdr_torch

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
        negative_loss_weight: float = 1.0,
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
        self.negative_loss_weight = negative_loss_weight


    def freeze_backbone(self):
        r"""Freeze the backbone (ResUNet and CLAP encoder) and only keep the 
        transformer bottleneck unfrozen for training.
        """
        # 1. Freeze query encoder
        for param in self.query_encoder.parameters():
            param.requires_grad = False
        
        # 2. Freeze SS model (backbone)
        for param in self.ss_model.parameters():
            param.requires_grad = False
            
        # 3. Unfreeze transformer bottleneck if it exists
        if hasattr(self.ss_model, 'base') and hasattr(self.ss_model.base, 'transformer_bottleneck'):
            for param in self.ss_model.base.transformer_bottleneck.parameters():
                param.requires_grad = True
            logging.info("Froze backbone and unfroze transformer bottleneck.")
        else:
            raise RuntimeError(
                "Transformer bottleneck not found in ss_model. "
                "Cannot train with --freeze_backbone if no transformer is present."
            )


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

        pre_mixed = 'mixture' in batch_audio_text_dict and 'segment' in batch_audio_text_dict
        if pre_mixed:
            mixtures = batch_audio_text_dict['mixture']
            segments = batch_audio_text_dict['segment']
        else:
            mixtures, segments = self.waveform_mixer(
                waveforms=batch_audio
            )

        # calculate text embed for audio-text data
        if self.query_encoder_type == 'CLAP':
            conditions = self.query_encoder.get_query_embed(
                modality='text' if pre_mixed else 'hybird',
                text=batch_text,
                audio=segments.squeeze(1),
                use_text_ratio=self.use_text_ratio,
            )

        input_dict = {
            'mixture': mixtures[:, None, :] if mixtures.ndim == 2 else mixtures,
            'condition': conditions,
        }

        target_dict = {
            'segment': segments.squeeze(1),
        }

        self.ss_model.train()
        sep_segment = self.ss_model(input_dict)['waveform']
        if sep_segment.ndim == 3 and sep_segment.size(1) == 1:
            sep_segment = sep_segment.squeeze(1)
        # (batch_size, 1, segment_samples)

        output_dict = {
            'segment': sep_segment,
        }

        # Calculate loss.
        if (
            pre_mixed
            and 'is_negative' in batch_audio_text_dict
            and self.negative_loss_weight != 1.0
        ):
            per_sample_loss = torch.mean(
                torch.abs(output_dict['segment'] - target_dict['segment']),
                dim=tuple(range(1, output_dict['segment'].ndim)),
            )
            weights = torch.where(
                batch_audio_text_dict['is_negative'].bool(),
                torch.full_like(per_sample_loss, float(self.negative_loss_weight)),
                torch.ones_like(per_sample_loss),
            )
            loss = torch.mean(per_sample_loss * weights)
        else:
            loss = self.loss_function(output_dict, target_dict)

        log_dict = {"train_loss": loss}
        if 'is_negative' in batch_audio_text_dict:
            log_dict["train_negative_ratio"] = batch_audio_text_dict['is_negative'].float().mean()
        if 'pair_similarity' in batch_audio_text_dict:
            log_dict["train_pair_similarity"] = batch_audio_text_dict['pair_similarity'].float().mean()
        if 'is_hard_pair' in batch_audio_text_dict:
            log_dict["train_hard_pair_ratio"] = batch_audio_text_dict['is_hard_pair'].float().mean()
        if 'is_easy_pair' in batch_audio_text_dict:
            log_dict["train_easy_pair_ratio"] = batch_audio_text_dict['is_easy_pair'].float().mean()
        if 'is_medium_pair' in batch_audio_text_dict:
            log_dict["train_medium_pair_ratio"] = batch_audio_text_dict['is_medium_pair'].float().mean()
        self.log_dict(log_dict)
        
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
            'mixture': mixtures[:, None, :] if mixtures.ndim == 2 else mixtures,
            'condition': conditions,
        }

        self.ss_model.eval()
        with torch.no_grad():
            sep_segment = self.ss_model(input_dict)['waveform']
            sep_segment = sep_segment.squeeze(1) # (batch, samples)

        # Ground truth and mixtures for comparison
        segments = segments.squeeze(1) # (batch, samples)
        mixtures = mixtures.squeeze(1) # (batch, samples)

        # Calculate metrics using stable torch implementations
        sdr = calculate_sdr_torch(ref=segments, est=sep_segment)
        sdr_no_sep = calculate_sdr_torch(ref=segments, est=mixtures)
        sdri = sdr - sdr_no_sep
        sisdr = calculate_sisdr_torch(ref=segments, est=sep_segment)

        # Log mean values across the batch
        self.log("val_sdr", sdr.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_sdri", sdri.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_sisdr", sisdr.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return sdr.mean()

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
