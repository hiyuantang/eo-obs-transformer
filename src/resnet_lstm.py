import torch
import torch.nn as nn
from typing import Dict, Any

from resnet_block import PyTorchResNetBlock
from lstm_block import PyTorchLstmBlock, PyTorchLstmBlockWithHead

class PyTorchResNetLstm:

    @staticmethod
    def get_model_with_head(model_params: Dict[str, Any]) -> nn.Module:
        """
        Builds the combined ResNet-LSTM model with a regression head.

        Args:
            model_params: Dictionary containing parameters for both
                          ResNetBlock and LstmBlock, including:
                          - ms_channels, path_to_ms_model, path_to_nl_model, etc. (for ResNet)
                          - n_of_frames, n_lstm_units, lstm_dropout, bidirectional (for LSTM)
                          - l2 (for LSTM head)
                          - name (optional, for potential model saving/loading)

        Returns:
            A PyTorch nn.Module representing the ResNet-LSTM model with head.
        """
        print("Initializing PyTorch ResNet Block...")
        res_block_params = model_params.copy()
        res_block_params['batch_norm_final'] = model_params.get('resnet_batch_norm_final', True)
        res_block = PyTorchResNetBlock(**res_block_params)

        print("Initializing PyTorch LSTM Block with Head...")
        lstm_params = model_params.copy()
        lstm_params['input_feature_dim'] = res_block.combined_feature_dim

        # Create the base LSTM block first
        base_lstm_block = PyTorchLstmBlock(input_block=res_block, **lstm_params)

        # Add the head to the LSTM block
        model = PyTorchLstmBlockWithHead(lstm_block=base_lstm_block, **lstm_params)

        # Store the name if provided (useful for saving/loading)
        model_name = model_params.get('name', 'ResNetLSTM_WithHead')
        print(f"Model Name: {model_name}")
        # You might want to attach the name to the model object if needed later
        # model.name = model_name

        return model

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':

    # Example parameters (adjust as needed)
    params = {
        'ms_channels': ['BLUE', 'GREEN', 'RED', 'NIR'], # Example: 4 MS channels
        'pretrained_ms': True,
        'pretrained_nl': False, # Example: NL from scratch
        'freeze_model': False,
        'resnet_batch_norm_final': True, # BN after concat in ResNet base

        'n_of_frames': 10,
        'n_lstm_units': 32,
        'lstm_dropout': 0.3,
        'bidirectional': True,
        'trainable_input_block': True, # Train ResNet part

        'l2': 0.05, # L2 for the final regression head (optimizer weight decay)
        'name': 'MyPyTorchResNetLSTM'
    }

    print("Building Model...")
    model = PyTorchResNetLstm.get_model_with_head(params)
    print("Model Built:")
    print(model)

    # --- Dummy Input Data ---
    batch_size = 4
    time_steps = params['n_of_frames']
    num_ms_channels = len(params['ms_channels'])
    num_nl_channels = 1 # Assuming 1 NL channel
    total_channels = num_ms_channels + num_nl_channels
    height, width = 224, 224

    # Input tensor (batch, time, channels, height, width)
    dummy_input = torch.randn(batch_size, time_steps, total_channels, height, width)
    # Optional mask (batch, time, 1) - e.g., mask last 2 steps
    dummy_mask = torch.ones(batch_size, time_steps, 1)
    if time_steps > 2 :
         dummy_mask[:, -2:, :] = 0

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Mask shape: {dummy_mask.shape}")


    # --- Forward Pass ---
    print("\nPerforming forward pass...")
    model.eval() # Set model to evaluation mode if not training
    with torch.no_grad(): # Disable gradient calculation for inference
        output = model(dummy_input, outputs_mask=dummy_mask)
    print(f"Output shape: {output.shape}") # Should be (batch, time, 1)

    # --- Check Output where mask is 0 ---
    if time_steps > 2 :
      print(f"Output for last 2 (masked) steps (should be 0):\n{output[:, -2:, :]}")
      print(f"Output for earlier (unmasked) steps:\n{output[:, :-2, :]}")