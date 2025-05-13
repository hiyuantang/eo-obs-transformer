import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
import numpy as np
import math 
from typing import List, Dict, Tuple, Optional
import csv 
from sat_dataset import TFRecordSenegalDataset
from resnet_lstm import PyTorchResNetLstm 

# --- Configuration ---
TFRECORD_DIR = 'downloaded_gcs_data/data_dhs'
CSV_PATH = 'data/senegal_shuffled.csv'
FRAME_YEARS = list(range(1990, 2018, 3))
N_FRAMES = len(FRAME_YEARS) 

TARGET_BANDS = [
    'BLUE', 'GREEN', 'RED',
    'NIR', 'SWIR1', 'SWIR2',
    'TEMP1',
    'NL'
]
IMAGE_SIZE = 224

MODEL_PARAMS = {
    # ResNetBlock params
    'ms_channels': [b for b in TARGET_BANDS if b != 'NL'],
    'pretrained_ms': True,
    'pretrained_nl': True,
    'freeze_model': False,
    'resnet_batch_norm_final': True,

    # LstmBlock params
    'n_of_frames': N_FRAMES, # Set explicitly to 10
    'n_lstm_units': 512,
    'lstm_dropout': 0.2,
    'bidirectional': True,

    # LstmBlockWithHead params
    'l2': 0.01, 
    'name': 'Senegal_ResNetLSTM_TargetYearMask' 
}

# Training Parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 25
RANDOM_SEED = 42
TRAIN_SPLIT_RATIO = 0.8
NUM_WORKERS = 0
LOG_DIR = 'log'
SAVE_EVERY_N_EPOCHS = 5


# --- Helper Functions ---

def get_device():
    """Detects and returns the best available device."""
    if torch.cuda.is_available():
        print("CUDA detected. Using GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        if torch.backends.mps.is_built():
             print("MPS detected and available. Using Apple Metal GPU.")
             return torch.device("mps")
        else:
            print("MPS device found but not built. Using CPU.")
            return torch.device("cpu")
    else:
        print("No GPU detected. Using CPU.")
        return torch.device("cpu")

def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for TFRecordSenegalDataset.
    - Uses fixed FRAME_YEARS [1990, 1993, ..., 2017].
    - Pads sequences to length N_FRAMES (10).
    - Creates a mask that is 1 only at the frame index corresponding
      to the sample's target_year, 0 otherwise.
    """
    batch_size = len(batch)
    valid_batch_items = [] 

    for item in batch:
        target_year = item.get('target_year')
        images_data = item.get('images_by_year', {})

        # Check if target_year is valid and falls within any frame's 3-year window
        if target_year is None or target_year < FRAME_YEARS[0] or target_year >= (FRAME_YEARS[-1] + 3):
            continue

        if not images_data:
            continue

        valid_batch_items.append(item)

    # If no valid items left in batch, return empty tensors
    if not valid_batch_items:
        return torch.empty(0), torch.empty(0), torch.empty(0)

    # --- Process valid items ---
    actual_batch_size = len(valid_batch_items)
    num_bands = len(TARGET_BANDS)
    img_h, img_w = IMAGE_SIZE, IMAGE_SIZE

    images_batch = torch.zeros(actual_batch_size, N_FRAMES, num_bands, img_h, img_w, dtype=torch.float32)
    labels_batch = torch.zeros(actual_batch_size, 1, dtype=torch.float32)
    outputs_mask_batch = torch.zeros(actual_batch_size, N_FRAMES, 1, dtype=torch.float32)

    for i, item in enumerate(valid_batch_items):
        images_by_year = item['images_by_year']
        target_year = item['target_year']
        iwi_label = item['iwi']

        labels_batch[i, 0] = iwi_label

        for frame_idx, frame_start_year in enumerate(FRAME_YEARS):
            if frame_start_year in images_by_year:
                images_batch[i, frame_idx, ...] = images_by_year[frame_start_year]

        target_frame_index = -1
        for fi, frame_start_year in enumerate(FRAME_YEARS):
            if frame_start_year <= target_year < frame_start_year + 3:
                target_frame_index = fi
                break 

        if target_frame_index != -1:
             outputs_mask_batch[i, target_frame_index, 0] = 1.0

    return images_batch, labels_batch, outputs_mask_batch


# --- Main Training Script ---
if __name__ == "__main__":
    print("--- Setup ---")
    device = get_device()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed(RANDOM_SEED)
    elif device.type == 'mps':
         pass

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"Created log directory: {LOG_DIR}")

    # <<< Added: Setup log file >>>
    log_file_path = os.path.join(LOG_DIR, 'training_log.csv')
    write_header = not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0
    with open(log_file_path, 'a', newline='') as f:
        log_writer = csv.writer(f)
        if write_header:
            log_writer.writerow(['Epoch', 'TrainLoss', 'ValLoss']) # Header

    # --- Load Dataset ---
    print("Loading dataset...")
    try:
        full_dataset = TFRecordSenegalDataset(
            tfrecord_dir=TFRECORD_DIR,
            csv_path=CSV_PATH,
            target_bands=TARGET_BANDS,
            image_size=IMAGE_SIZE
        )
        print(f"Dataset loaded successfully. Found {len(full_dataset)} samples.")
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure TFRECORD_DIR and CSV_PATH are correct.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading dataset: {e}")
        exit()

    if len(full_dataset) == 0:
        print("Error: Dataset is empty after loading.")
        exit()

    # --- Split Dataset ---
    print("Splitting dataset...")
    total_size = len(full_dataset)
    train_size = int(TRAIN_SPLIT_RATIO * total_size)
    val_size = total_size - train_size
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- Create DataLoaders ---
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
    )

    # --- Initialize Model ---
    print("Initializing model...")
    MODEL_PARAMS['n_of_frames'] = N_FRAMES
    model = PyTorchResNetLstm.get_model_with_head(MODEL_PARAMS) 
    model.to(device)

    # --- Loss and Optimizer ---
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=MODEL_PARAMS.get('l2', 0.0))

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # ===================================
        # Training Phase
        # ===================================
        model.train()
        train_loss_accum_sum = 0.0  
        total_unmasked_train = 0   

        train_pbar = tqdm(train_loader, desc=f"Train E{epoch+1}", leave=False)
        for i, (images, labels, masks) in enumerate(train_pbar):
            if images.numel() == 0 or labels.numel() == 0 or masks.numel() == 0:
                continue
            images, labels, masks = images.to(device), labels.to(device), masks.to(device, dtype=torch.bool)

            optimizer.zero_grad()
            outputs = model(images, outputs_mask=masks)
            expanded_labels = labels.unsqueeze(1).expand(-1, N_FRAMES, -1).float()
            per_element_loss = criterion(outputs, expanded_labels)
            masked_per_element_loss = per_element_loss.masked_fill(~masks, 0.0)
            total_loss_batch = masked_per_element_loss.sum()
            num_unmasked = masks.sum()

            if num_unmasked.item() > 0: 
                final_average_loss = total_loss_batch / num_unmasked
                final_average_loss.backward()
                optimizer.step()

                train_loss_accum_sum += total_loss_batch.item() 
                total_unmasked_train += num_unmasked.item()     
                train_pbar.set_postfix(loss=f"{final_average_loss.item():.4f}")
            else:
                train_pbar.set_postfix(loss="N/A (all masked)")

        # Calculate average training loss for the epoch
        avg_train_loss = train_loss_accum_sum / total_unmasked_train if total_unmasked_train > 0 else 0.0
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")


        # ===================================
        # Validation Phase
        # ===================================
        model.eval()
        val_loss_accum_sum = 0.0  
        total_unmasked_val = 0      

        val_pbar = tqdm(val_loader, desc=f"Val E{epoch+1}", leave=False)
        with torch.no_grad(): 
            for images, labels, masks in val_pbar:
                if images.numel() == 0 or labels.numel() == 0 or masks.numel() == 0:
                    continue
                images, labels, masks = images.to(device), labels.to(device), masks.to(device, dtype=torch.bool)
                outputs = model(images, outputs_mask=masks)
                expanded_labels = labels.unsqueeze(1).expand(-1, N_FRAMES, -1).float()
                per_element_loss = criterion(outputs, expanded_labels) 
                masked_per_element_loss = per_element_loss.masked_fill(~masks, 0.0)

                total_loss_batch = masked_per_element_loss.sum()
                num_unmasked = masks.sum()

                if num_unmasked.item() > 0:
                    batch_average_loss = total_loss_batch / num_unmasked
                    val_loss_accum_sum += total_loss_batch.item() 
                    total_unmasked_val += num_unmasked.item()    


                    val_pbar.set_postfix(loss=f"{batch_average_loss.item():.4f}")
                else:
                    val_pbar.set_postfix(loss="N/A (all masked)")

        avg_val_loss = val_loss_accum_sum / total_unmasked_val if total_unmasked_val > 0 else 0.0
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

        with open(log_file_path, 'a', newline='') as f:
             log_writer = csv.writer(f)
             log_writer.writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}"]) 


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            model_save_name = f"{MODEL_PARAMS.get('name', 'resnet_lstm')}_best.pth" 
            model_save_path = os.path.join(LOG_DIR, model_save_name)
            print(f"Validation loss improved. Saving best model to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
        
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_save_name = f"{MODEL_PARAMS.get('name', 'resnet_lstm')}_epoch_{epoch+1}.pth"
            checkpoint_save_path = os.path.join(LOG_DIR, checkpoint_save_name)
            print(f"Saving checkpoint for epoch {epoch+1} to {checkpoint_save_path}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss, 
            }, checkpoint_save_path)

    print("\n--- Training Finished ---")
    print(f"Best Validation Loss recorded: {best_val_loss:.4f}")
    print(f"Training log saved to: {log_file_path}")
    best_model_path = os.path.join(LOG_DIR, f"{MODEL_PARAMS.get('name', 'resnet_lstm')}_best.pth")
    if os.path.exists(best_model_path):
        print(f"Best model saved to: {best_model_path}")
    else:
        print("No model was saved (possibly validation loss did not improve).")