import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps = x.shape[0], x.shape[1]
        # Combine batch and time dimensions
        # Input shape: (batch, time, C, H, W) -> (batch * time, C, H, W)
        # Input shape: (batch, time, features) -> (batch * time, features)
        input_reshaped = x.contiguous().view(batch_size * time_steps, *x.shape[2:])

        output_reshaped = self.module(input_reshaped)

        # Reshape output back to (batch, time, ...)
        # Output shape: (batch * time, out_features) -> (batch, time, out_features)
        # Output shape: (batch * time, C', H', W') -> (batch, time, C', H', W')
        output_original_shape = output_reshaped.view(batch_size, time_steps, *output_reshaped.shape[1:])

        return output_original_shape

class PyTorchLstmBlock(nn.Module):
    def __init__(self,
                 input_block: nn.Module,
                 n_of_frames: int, # Primarily needed for defining mask shape if using head
                 input_feature_dim: int, # The feature dimension output by input_block for one frame
                 n_lstm_units: int = 18,
                 lstm_dropout: float = 0.2,
                 bidirectional: bool = True,
                 **kwargs: Any):
        super().__init__()

        # It's crucial that input_block processes a single frame/image
        # Input: (batch * time, C, H, W) -> Output: (batch * time, input_feature_dim)
        self.input_block = input_block
        self.n_of_frames = n_of_frames # Store for potential use (e.g., mask creation)

        # Apply the input block over the time dimension
        self.time_distributed_input = TimeDistributed(self.input_block)

        self.lstm = nn.LSTM(
            input_size=input_feature_dim,
            hidden_size=n_lstm_units,
            num_layers=1, 
            batch_first=True, # Expect input as (batch, seq_len, features)
            dropout=lstm_dropout if lstm_dropout > 0 else 0, # Keras LSTM dropout is recurrent dropout. PyTorch dropout applies to outputs of layers except last. They are different! 
            bidirectional=bidirectional
        )

        # Calculate LSTM output feature size
        self.lstm_output_features = n_lstm_units * 2 if bidirectional else n_lstm_units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM block (without head).

        Args:
            x: Input tensor of shape (batch_size, time_steps, C, H, W)
               where time_steps == n_of_frames.

        Returns:
            Output tensor of shape (batch_size, time_steps, lstm_output_features).
               Contains LSTM outputs for each time step.
        """
        # Apply input block across time
        # x: (batch, time, C, H, W)
        features = self.time_distributed_input(x)
        # features: (batch, time, input_feature_dim)

        # Apply LSTM
        # LSTM expects input: (batch, seq_len, input_size)
        # LSTM outputs: (output, (h_n, c_n))
        # output shape: (batch, seq_len, num_directions * hidden_size)
        lstm_outputs, _ = self.lstm(features)
        # lstm_outputs: (batch, time, lstm_output_features)

        return lstm_outputs

class PyTorchLstmBlockWithHead(nn.Module):
    """ LSTM block combined with a regression head and masking. """
    def __init__(self, lstm_block: PyTorchLstmBlock, l2: float = 0.01, **kwargs: Any):
        super().__init__()
        self.lstm_block = lstm_block
        self.n_of_frames = lstm_block.n_of_frames # Get from underlying block

        lstm_output_features = lstm_block.lstm_output_features

        # Dense layer applied time-wise
        self.dense = nn.Linear(lstm_output_features, 1)
        # Apply Dense layer across time dimension
        self.time_distributed_dense = TimeDistributed(self.dense)

        # Activation applied time-wise
        self.activation = nn.ReLU()
        self.time_distributed_activation = TimeDistributed(self.activation)

        self.l2 = l2 # Store for optimizer weight_decay reference
        print(f"LSTM Head L2 (weight decay in optimizer): {self.l2}")

    def forward(self, x: torch.Tensor, outputs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with regression head and optional masking.

        Args:
            x: Input tensor for the LSTM block
               (batch_size, time_steps, C, H, W).
            outputs_mask: Optional mask tensor of shape (batch_size, time_steps, 1)
                          or broadcastable shape. Values should be 0 or 1.

        Returns:
            Output tensor of shape (batch_size, time_steps, 1), potentially masked.
        """
        # Get LSTM sequence outputs
        lstm_outputs = self.lstm_block(x) # (batch, time, lstm_features)

        # Apply dense layer across time
        dense_logits = self.time_distributed_dense(lstm_outputs) # (batch, time, 1)

        # Apply activation across time
        unmasked_outputs = self.time_distributed_activation(dense_logits) # (batch, time, 1)

        # Apply mask if provided
        if outputs_mask is not None:
            # Ensure mask has compatible dimensions for broadcasting
            if outputs_mask.dim() == 2: # e.g., (batch, time) -> (batch, time, 1)
                 outputs_mask = outputs_mask.unsqueeze(-1)
            # Element-wise multiplication
            masked_outputs = unmasked_outputs * outputs_mask
            return masked_outputs
        else:
            # Return unmasked outputs if no mask is given
            return unmasked_outputs