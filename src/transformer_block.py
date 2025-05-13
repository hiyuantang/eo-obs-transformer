# transformer_block.py
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from typing import Optional, Tuple, Any

# Assuming TimeDistributed is defined here or imported
class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps = x.shape[0], x.shape[1]
        input_reshaped = x.contiguous().view(batch_size * time_steps, *x.shape[2:])
        output_reshaped = self.module(input_reshaped)
        output_original_shape = output_reshaped.view(batch_size, time_steps, *output_reshaped.shape[1:])
        return output_original_shape

# Helper class for Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50): # Increased max_len
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # Check if d_model is odd, handle the last dimension if so
        if d_model % 2 != 0:
             # Use the same calculation as the previous even dimension for the last odd one
             pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:d_model//2]
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe) # shape (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x shape needs to be (seq_len, batch, features) before adding pe
        # self.pe shape is (max_len, 1, d_model)
        # We need to select the part of pe that matches the seq_len of x
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PyTorchTransformerEncoderBlock(nn.Module):
    def __init__(self,
                 input_block: nn.Module,
                 n_of_frames: int,
                 input_feature_dim: int, # Feature dim output by input_block for one frame
                 d_model: int = 128,     # Embedding dimension for Transformer (must match input_feature_dim)
                 nhead: int = 8,         # Number of attention heads
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 256,
                 transformer_dropout: float = 0.1,
                 pos_dropout: float = 0.1,
                 activation: str = "relu",
                 **kwargs: Any):
        super().__init__()

        if d_model != input_feature_dim:
             print(f"Warning: d_model ({d_model}) != input_feature_dim ({input_feature_dim}). "
                   f"Transformer performance may be affected or errors may occur "
                   f"if input_feature_dim is not the intended embedding dimension.")
             # If you want the transformer to work with a different dimension,
             # you might need a linear layer here to project input_feature_dim to d_model.
             # For simplicity now, we assume they are meant to be the same.
             # If they MUST differ, add: self.input_proj = nn.Linear(input_feature_dim, d_model)
             # and apply it after time_distributed_input.
             # Let's enforce they are the same for now for simplicity matching LSTM structure.
             if 'input_proj' not in kwargs or not kwargs['input_proj']: # Allow override via kwargs if needed
                 print(f"Adjusting d_model to match input_feature_dim: {input_feature_dim}")
                 d_model = input_feature_dim
             else:
                 print("Proceeding with different d_model and input_feature_dim. Ensure projection layer handles this.")


        # Store parameters
        self.input_block = input_block
        self.n_of_frames = n_of_frames
        self.d_model = d_model # This is the expected feature dimension for the transformer layers

        # Apply the input block over the time dimension
        self.time_distributed_input = TimeDistributed(self.input_block)

        # Optional projection layer if input_feature_dim != d_model
        if input_feature_dim != self.d_model:
            self.input_proj = nn.Linear(input_feature_dim, self.d_model)
        else:
            self.input_proj = nn.Identity() # No projection needed

        # Positional Encoding
        # n_of_frames corresponds to the sequence length
        self.pos_encoder = PositionalEncoding(self.d_model, pos_dropout, max_len=n_of_frames + 1) # max_len > n_of_frames

        # Transformer Encoder Layer
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation=activation,
            batch_first=False # TransformerEncoder expects (Seq, Batch, Feature)
        )

        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # Output feature dimension is d_model
        self.transformer_output_features = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer Encoder block (without head).

        Args:
            x: Input tensor of shape (batch_size, time_steps, C, H, W)
               where time_steps == n_of_frames.

        Returns:
            Output tensor of shape (batch_size, time_steps, d_model).
               Contains Transformer outputs for each time step.
        """
        # Apply input block across time
        # x: (batch, time, C, H, W)
        features = self.time_distributed_input(x)
        # features: (batch, time, input_feature_dim)

        # Apply projection if needed
        features = self.input_proj(features) # (batch, time, d_model)

        # Transformer expects (Seq, Batch, Feature) by default
        # features shape: (batch, time, d_model) -> (time, batch, d_model)
        features_transposed = features.permute(1, 0, 2)

        # Add positional encoding
        features_pos_encoded = self.pos_encoder(features_transposed)
        # features_pos_encoded shape: (time, batch, d_model)

        # Apply Transformer Encoder
        transformer_outputs = self.transformer_encoder(features_pos_encoded)
        # transformer_outputs shape: (time, batch, d_model)

        # Permute back to (Batch, Seq, Feature) for consistency with LSTM block output
        # output shape: (time, batch, d_model) -> (batch, time, d_model)
        transformer_outputs_final = transformer_outputs.permute(1, 0, 2)

        return transformer_outputs_final

class PyTorchTransformerEncoderBlockWithHead(nn.Module):
    """ Transformer Encoder block combined with a regression head and masking. """
    def __init__(self, transformer_block: PyTorchTransformerEncoderBlock, l2: float = 0.01, **kwargs: Any):
        super().__init__()
        self.transformer_block = transformer_block
        self.n_of_frames = transformer_block.n_of_frames

        transformer_output_features = transformer_block.transformer_output_features

        # Dense layer applied time-wise
        self.dense = nn.Linear(transformer_output_features, 1)
        # Apply Dense layer across time dimension
        self.time_distributed_dense = TimeDistributed(self.dense)

        # Activation applied time-wise
        self.activation = nn.ReLU()
        self.time_distributed_activation = TimeDistributed(self.activation)

        self.l2 = l2 # Store for optimizer weight_decay reference
        print(f"Transformer Head L2 (weight decay in optimizer): {self.l2}")

    def forward(self, x: torch.Tensor, outputs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with regression head and optional masking.

        Args:
            x: Input tensor for the Transformer block
               (batch_size, time_steps, C, H, W).
            outputs_mask: Optional mask tensor of shape (batch_size, time_steps, 1)
                          or broadcastable shape. Values should be 0 or 1 (float or bool).

        Returns:
            Output tensor of shape (batch_size, time_steps, 1), potentially masked.
        """
        # Get Transformer sequence outputs
        transformer_outputs = self.transformer_block(x) # (batch, time, d_model)

        # Apply dense layer across time
        dense_logits = self.time_distributed_dense(transformer_outputs) # (batch, time, 1)

        # Apply activation across time
        unmasked_outputs = self.time_distributed_activation(dense_logits) # (batch, time, 1)

        # Apply mask if provided
        if outputs_mask is not None:
            # Ensure mask has compatible dimensions and dtype
            if outputs_mask.dim() == 2: # e.g., (batch, time) -> (batch, time, 1)
                 outputs_mask = outputs_mask.unsqueeze(-1)
            if not isinstance(outputs_mask, torch.FloatTensor) and not isinstance(outputs_mask, torch.cuda.FloatTensor):
                 # Ensure mask is float for multiplication if it's boolean or int
                 outputs_mask = outputs_mask.float()

            # Element-wise multiplication
            masked_outputs = unmasked_outputs * outputs_mask
            return masked_outputs
        else:
            # Return unmasked outputs if no mask is given
            return unmasked_outputs