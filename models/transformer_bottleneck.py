import torch
import torch.nn as nn
from models.base import init_layer


class TransformerBottleneck(nn.Module):
    """Cross-attention transformer bottleneck for text-conditioned audio separation.

    Replaces the convolutional bottleneck (conv_block7a) in ResUNet30_Base.
    Audio tokens serve as queries; the CLAP text embedding serves as key/value.
    This is inspired by the DPRNN bottleneck in "Language-Queried Audio Source
    Separation via ResUNet with DPRNN" (DCASE 2024), substituting DPRNN with
    a cross-attention transformer.

    Args:
        audio_channels: Channel dimension of the encoder output (default: 384).
        text_embed_dim: Dimension of the CLAP text embedding (default: 512).
        d_model: Transformer hidden dimension (default: 384).
        nhead: Number of attention heads (default: 8).
        num_layers: Number of transformer decoder layers (default: 4).
        dim_feedforward: FFN intermediate dimension (default: 1536).
        dropout: Dropout rate (default: 0.1).
        max_h: Maximum spatial height for positional encoding (default: 32).
        max_w: Maximum spatial width for positional encoding (default: 32).
    """

    def __init__(
        self,
        audio_channels: int = 384,
        text_embed_dim: int = 512,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        max_h: int = 32,
        max_w: int = 32,
    ):
        super(TransformerBottleneck, self).__init__()

        self.d_model = d_model
        self.audio_channels = audio_channels

        # Project CLAP text embedding into transformer dimension
        self.text_proj = nn.Linear(text_embed_dim, d_model)
        init_layer(self.text_proj)

        # Project audio channels to d_model if they differ
        if audio_channels != d_model:
            self.audio_input_proj = nn.Linear(audio_channels, d_model)
            self.audio_output_proj = nn.Linear(d_model, audio_channels)
            init_layer(self.audio_input_proj)
            init_layer(self.audio_output_proj)
        else:
            self.audio_input_proj = None
            self.audio_output_proj = None

        # 2D learnable positional encodings (height and width, concatenated)
        self.pos_encoding_h = nn.Embedding(max_h, d_model // 2)
        self.pos_encoding_w = nn.Embedding(max_w, d_model // 2)

        # Cross-attention transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_layers = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )

        # Final layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def _get_2d_pos_encoding(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Generate 2D positional encoding by concatenating height and width embeddings.

        Args:
            h: Spatial height.
            w: Spatial width.
            device: Target device.

        Returns:
            pos: (h*w, d_model) positional encoding tensor.
        """
        h_indices = torch.arange(h, device=device)
        w_indices = torch.arange(w, device=device)

        h_embed = self.pos_encoding_h(h_indices)  # (h, d_model//2)
        w_embed = self.pos_encoding_w(w_indices)  # (w, d_model//2)

        # Broadcast and concatenate: each (h, w) position gets [h_embed || w_embed]
        h_embed = h_embed.unsqueeze(1).expand(-1, w, -1)  # (h, w, d_model//2)
        w_embed = w_embed.unsqueeze(0).expand(h, -1, -1)  # (h, w, d_model//2)

        pos = torch.cat([h_embed, w_embed], dim=-1)  # (h, w, d_model)
        pos = pos.reshape(h * w, self.d_model)  # (h*w, d_model)

        return pos

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer bottleneck.

        Args:
            x: Audio features from the last encoder block, shape (B, C, H, W).
            condition: CLAP text embedding, shape (B, text_embed_dim).

        Returns:
            out: Transformed audio features, shape (B, C, H, W) — same as input.
        """
        B, C, H, W = x.shape

        # 1. Flatten spatial dims: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # 2. Project audio channels to d_model if needed
        if self.audio_input_proj is not None:
            x_flat = self.audio_input_proj(x_flat)  # (B, H*W, d_model)

        # 3. Add 2D positional encoding
        pos = self._get_2d_pos_encoding(H, W, x.device)  # (H*W, d_model)
        x_flat = x_flat + pos.unsqueeze(0)  # broadcast over batch

        # 4. Project text embedding: (B, text_embed_dim) -> (B, 1, d_model)
        text_tokens = self.text_proj(condition).unsqueeze(1)

        # 5. Cross-attention: Q=audio_tokens, K/V=text_token
        x_flat = self.cross_attn_layers(
            tgt=x_flat,       # (B, H*W, d_model) — audio as query
            memory=text_tokens,  # (B, 1, d_model) — text as key/value
        )

        # 6. Final layer norm
        x_flat = self.layer_norm(x_flat)

        # 7. Project back to audio channels if needed
        if self.audio_output_proj is not None:
            x_flat = self.audio_output_proj(x_flat)  # (B, H*W, C)

        # 8. Reshape back to spatial: (B, H*W, C) -> (B, C, H, W)
        out = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return out
