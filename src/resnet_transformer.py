import torch
import torch.nn as nn
from typing import Dict, Any
from resnet_block import PyTorchResNetBlock #
from transformer_block import PyTorchTransformerEncoderBlock, PyTorchTransformerEncoderBlockWithHead

class PyTorchResNetTransformer:

    @staticmethod
    def get_model_with_head(model_params: Dict[str, Any]) -> nn.Module:
        """
        Builds the combined ResNet-Transformer model with a regression head.

        Args:
            model_params: Dictionary containing parameters for both
                          ResNetBlock and TransformerEncoderBlock, including:
                          - ms_channels, etc. (for ResNet)
                          - n_of_frames, d_model, nhead, num_encoder_layers,
                            transformer_dropout, pos_dropout, activation (for Transformer)
                          - l2 (for Transformer head)
                          - name (optional)

        Returns:
            A PyTorch nn.Module representing the ResNet-Transformer model with head.
        """
        print("Initializing PyTorch ResNet Block...")
        res_block_params = model_params.copy()
        # Ensure ResNet output BN is handled (can be controlled via model_params)
        res_block_params['batch_norm_final'] = model_params.get('resnet_batch_norm_final', True)
        res_block = PyTorchResNetBlock(**res_block_params) #

        print("Initializing PyTorch Transformer Encoder Block with Head...")
        transformer_params = model_params.copy()
        # The input feature dimension for the transformer is the output of the ResNet block
        transformer_params['input_feature_dim'] = res_block.combined_feature_dim

        # Create the base Transformer Encoder block first
        # Pass relevant transformer parameters from model_params
        base_transformer_block = PyTorchTransformerEncoderBlock(
            input_block=res_block,
            **transformer_params # Pass all relevant params like n_of_frames, d_model, nhead etc.
        )

        # Add the head to the Transformer block
        model = PyTorchTransformerEncoderBlockWithHead(
            transformer_block=base_transformer_block,
             # Pass head-specific params like l2
            **transformer_params
        )

        # Store the name if provided
        model_name = model_params.get('name', 'ResNetTransformer_WithHead')
        print(f"Model Name: {model_name}")
        # model.name = model_name # Optional: attach name to model object

        return model

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':

    # Example parameters for ResNetTransformer
    params = {
        # ResNetBlock params
        'ms_channels': ['BLUE', 'GREEN', 'RED', 'NIR'], # Example: 4 MS channels
        'pretrained_ms': True,
        'pretrained_nl': False,
        'freeze_model': False,
        'resnet_batch_norm_final': True,

        # TransformerEncoderBlock params
        'n_of_frames': 10,
        #'d_model': 1024, # This will be adjusted to match ResNet output (1024) if different
                          # Or you can add a projection layer by setting 'input_proj': True
        'nhead': 8,       # Must be divisor of d_model (adjusted input_feature_dim)
        'num_encoder_layers': 4,
        'dim_feedforward': 2048,
        'transformer_dropout': 0.1,
        'pos_dropout': 0.1,
        'activation': 'relu',
        #'input_proj': False, # Set to True if you want d_model != resnet output dim

        # TransformerEncoderBlockWithHead params
        'l2': 0.02, # L2 for the final regression head (optimizer weight decay)
        'name': 'MyPyTorchResNetTransformer'
    }

    print("Building Model...")
    model = PyTorchResNetTransformer.get_model_with_head(params)
    print("Model Built:")
    # print(model) # Can be very verbose

    # --- Dummy Input Data ---
    batch_size = 4
    time_steps = params['n_of_frames']
    num_ms_channels = len(params['ms_channels'])
    num_nl_channels = 1 # Assuming 1 NL channel
    total_channels = num_ms_channels + num_nl_channels
    # Get image size from ResNet block definition
    try:
        height, width = PyTorchResNetBlock.IMAGE_SIZE, PyTorchResNetBlock.IMAGE_SIZE # Use class variable
    except AttributeError:
        height, width = 224, 224 # Fallback if not defined directly on class

    # Input tensor (batch, time, channels, height, width)
    dummy_input = torch.randn(batch_size, time_steps, total_channels, height, width)
    # Optional mask (batch, time, 1) - e.g., mask last 2 steps
    dummy_mask = torch.ones(batch_size, time_steps, 1, dtype=torch.float32) # Use float mask
    if time_steps > 2 :
         dummy_mask[:, -2:, :] = 0.0

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