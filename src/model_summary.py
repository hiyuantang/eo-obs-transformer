import torch
from torchinfo import summary
import os
import traceback 

try:
    from resnet_lstm import PyTorchResNetLstm
    from resnet_transformer import PyTorchResNetTransformer
    models_available = True
except ImportError as e:
    print(f"Warning: Could not import model classes: {e}")
    print("Please ensure 'resnet_lstm.py' and 'resnet_transformer.py' are accessible.")
    models_available = False

# --- Common Configuration (derived from training scripts) ---
FRAME_YEARS = list(range(1990, 2018, 3)) # [1990, 1993, ..., 2017]
N_FRAMES = len(FRAME_YEARS) # Should be 10
TARGET_BANDS = [
    'BLUE', 'GREEN', 'RED',
    'NIR', 'SWIR1', 'SWIR2',
    'TEMP1',
    'NL'
]
NUM_BANDS = len(TARGET_BANDS)
IMAGE_SIZE = 224
BATCH_SIZE_FOR_SUMMARY = 1 # Use a batch size of 1 for summary display

# --- LSTM Model Parameters (from train_lstm.py) ---
LSTM_MODEL_PARAMS = {
    # ResNetBlock params
    'ms_channels': [b for b in TARGET_BANDS if b != 'NL'],
    'pretrained_ms': True,
    'pretrained_nl': True,
    'freeze_model': False,
    'resnet_batch_norm_final': True,

    # LstmBlock params
    'n_of_frames': N_FRAMES,
    'n_lstm_units': 512,
    'lstm_dropout': 0.3,
    'bidirectional': True,

    # LstmBlockWithHead params
    'l2': 0.01,

    # Optional name
    'name': 'Senegal_ResNetLSTM_TargetYearMask'
}

# --- Transformer Model Parameters (from train_trans.py) ---
TRANSFORMER_MODEL_PARAMS = {
    # ResNetBlock params
    'ms_channels': [b for b in TARGET_BANDS if b != 'NL'],
    'pretrained_ms': True,
    'pretrained_nl': True,
    'freeze_model': False,
    'resnet_batch_norm_final': True,

    # TransformerEncoderBlock params
    'n_of_frames': N_FRAMES,
    'd_model': 1024, 
    'nhead': 8,
    'num_encoder_layers': 1,
    'dim_feedforward': 1024, 
    'transformer_dropout': 0.1,
    'pos_dropout': 0.1,
    'activation': 'relu',

    # TransformerEncoderBlockWithHead params
    'l2': 0.01,
    'name': 'Senegal_ResNetTransformer_TargetYearMask'
}

# --- Define Input Shape ---
# Based on collate_fn: (B, N_FRAMES, num_bands, img_h, img_w)
input_shape = (BATCH_SIZE_FOR_SUMMARY, N_FRAMES, NUM_BANDS, IMAGE_SIZE, IMAGE_SIZE)


if __name__ == "__main__":
    if not models_available:
        print("Exiting due to missing model class definitions.")
        exit()

    # --- Get Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "="*50)
    print(" ResNet-LSTM Model Summary")
    print("="*50)
    try:
        # Instantiate LSTM model
        lstm_model = PyTorchResNetLstm.get_model_with_head(LSTM_MODEL_PARAMS)
        lstm_model.to(device)
        lstm_model.eval() # Set to evaluation mode

        # Create dummy input tensor for images ONLY
        dummy_images = torch.randn(*input_shape, device=device)

        print(f"Attempting summary for LSTM model with input shape: {dummy_images.shape}")
        # Pass ONLY the images tensor to summary
        summary(lstm_model,
                input_data=dummy_images, # <<< Only pass images
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                depth=4 # Adjust depth as needed
               )

    except Exception as e:
        print(f"Error processing ResNet-LSTM model: {e}")
        print("Traceback:")
        traceback.print_exc() # Print detailed traceback


    print("\n" + "="*50)
    print(" ResNet-Transformer Model Summary")
    print("="*50)
    try:
        # Instantiate Transformer model
        transformer_model = PyTorchResNetTransformer.get_model_with_head(TRANSFORMER_MODEL_PARAMS)
        transformer_model.to(device)
        transformer_model.eval() # Set to evaluation mode

        # Create dummy input tensor for images ONLY
        dummy_images = torch.randn(*input_shape, device=device)

        print(f"Attempting summary for Transformer model with input shape: {dummy_images.shape}")
        # Pass ONLY the images tensor to summary
        summary(transformer_model,
                input_data=dummy_images, # <<< Only pass images
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                depth=4 # Adjust depth as needed
               )

    except Exception as e:
        print(f"Error processing ResNet-Transformer model: {e}")
        print("Traceback:")
        traceback.print_exc() # Print detailed traceback

    print("\n--- Model Summaries Finished ---")