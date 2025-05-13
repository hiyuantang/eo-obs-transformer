import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score
from typing import List, Dict, Tuple, Optional
from sat_dataset import TFRecordSenegalDataset
from resnet_lstm import PyTorchResNetLstm
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR') 
import warnings
warnings.filterwarnings('ignore', message=r'.*tf.io.gfile.Glob.*')
from collections import defaultdict


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
    'ms_channels': [b for b in TARGET_BANDS if b != 'NL'],
    'pretrained_ms': True,
    'pretrained_nl': True,
    'freeze_model': False, 
    'resnet_batch_norm_final': True,
    'n_of_frames': N_FRAMES,
    'n_lstm_units': 512,
    'lstm_dropout': 0.2, 
    'bidirectional': True,
    'l2': 0.01, 
    'name': 'Senegal_ResNetLSTM_TargetYearMask'
}
BATCH_SIZE = 16 
RANDOM_SEED = 42
TRAIN_SPLIT_RATIO = 0.8
NUM_WORKERS = 0 


def get_device():
    """Detects and returns the best available device."""
    if torch.cuda.is_available():
        print("CUDA detected. Using GPU.")
        return torch.device("cuda")
    else:
        print("No GPU detected. Using CPU.")
        return torch.device("cpu")

def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(batch)
    valid_batch_items = []
    for item in batch:
        target_year = item.get('target_year')
        images_data = item.get('images_by_year', {})
        if target_year is None or target_year < FRAME_YEARS[0] or target_year >= (FRAME_YEARS[-1] + 3):
            continue
        if not images_data:
            continue
        valid_batch_items.append(item)

    if not valid_batch_items:
        return torch.empty(0), torch.empty(0), torch.empty(0)

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


def compute_r2(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """
    Computes the R-squared score for a model on a given dataset.

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader for the dataset (train or val).
        device: The device to run computations on (e.g., 'cuda', 'cpu').

    Returns:
        The R-squared score.
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, labels, masks in pbar:
            if images.numel() == 0: continue 

            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device, dtype=torch.bool) 

            outputs = model(images, outputs_mask=masks) 

            for i in range(outputs.shape[0]):
                mask_indices = torch.where(masks[i, :, 0])[0]
                if len(mask_indices) == 1:
                    valid_index = mask_indices[0]
                    pred = outputs[i, valid_index, 0].item()
                    label = labels[i, 0].item()
                    all_preds.append(pred)
                    all_labels.append(label)

    if not all_labels or not all_preds:
        print("Warning: No valid predictions/labels found during evaluation.")
        return 0.0

    r2 = r2_score(all_labels, all_preds)
    return r2

def main(model_path: str):
    """
    Loads data, model, computes and prints R^2 scores.
    """
    print("--- Setup ---")
    device = get_device()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed(RANDOM_SEED)

    print("Loading dataset...")
    try:
        full_dataset = TFRecordSenegalDataset(
            tfrecord_dir=TFRECORD_DIR,
            csv_path=CSV_PATH,
            target_bands=TARGET_BANDS,
            image_size=IMAGE_SIZE
        )
        print(f"Dataset loaded. Found {len(full_dataset)} samples.")
    except FileNotFoundError as e:
        print(f"Error: Dataset file/directory not found: {e}")
        print(f"Checked TFRECORD_DIR: {os.path.abspath(TFRECORD_DIR)}")
        print(f"Checked CSV_PATH: {os.path.abspath(CSV_PATH)}")
        exit()
    except Exception as e:
         print(f"An unexpected error occurred loading dataset: {e}")
         exit()

    if len(full_dataset) == 0:
        print("Error: Dataset is empty after loading.")
        exit()

    print("Splitting dataset...")
    total_size = len(full_dataset)
    train_size = int(TRAIN_SPLIT_RATIO * total_size)
    val_size = total_size - train_size
    if train_size == 0 or val_size == 0:
        print(f"Warning: Train ({train_size}) or Validation ({val_size}) split size is zero. Adjust ratio or check dataset size.")
        print("Evaluating on the full dataset instead.")
        train_dataset = full_dataset
        val_dataset = full_dataset 
    else:
        generator = torch.Generator().manual_seed(RANDOM_SEED)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn, 
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = None
    if len(val_dataset) > 0 and val_size > 0 : 
       val_loader = DataLoader(
           val_dataset,
           batch_size=BATCH_SIZE,
           shuffle=False,
           num_workers=NUM_WORKERS,
           collate_fn=collate_fn, 
           pin_memory=True if device.type == 'cuda' else False
       )


    print("Initializing model architecture...")
    model = PyTorchResNetLstm.get_model_with_head(MODEL_PARAMS)

    print(f"Loading trained model weights from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        exit()

    model.to(device)

    print("\n--- Computing R^2 Score ---")

    print("Calculating R^2 on Training Set...")
    train_r2 = compute_r2(model, train_loader, device)
    print(f"Training Set R^2: {train_r2:.4f}")

    if val_loader:
        print("\nCalculating R^2 on Validation Set...")
        val_r2 = compute_r2(model, val_loader, device)
        print(f"Validation Set R^2: {val_r2:.4f}")
    else:
        print("\nSkipping R^2 calculation on Validation Set (no validation samples).")

    print("\n--- Evaluation Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained ResNet-LSTM model (standalone).")
    parser.add_argument("model_path", type=str, help="Path to the trained model (.pth file).")
    args = parser.parse_args()
    if not os.path.isdir(TFRECORD_DIR):
        print(f"Error: TFRecord directory not found: {os.path.abspath(TFRECORD_DIR)}")
        exit()
    if not os.path.isfile(CSV_PATH):
        print(f"Error: CSV file not found: {os.path.abspath(CSV_PATH)}")
        exit()

    main(args.model_path)