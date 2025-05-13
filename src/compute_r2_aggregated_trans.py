import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score
from typing import List, Dict, Tuple, Optional
from sat_dataset import TFRecordSenegalDataset
from resnet_transformer import PyTorchResNetTransformer 
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings('ignore', message=r'.*tf.io.gfile.Glob.*')
from collections import defaultdict
import math 


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
    'nhead': 8,
    'num_encoder_layers': 1, 
    'dim_feedforward': 2048,
    'transformer_dropout': 0.1,
    'pos_dropout': 0.1,
    'activation': 'relu', 
    'l2': 0.01,
    'name': 'Senegal_ResNetTransformer_TargetYearMask' 
}
BATCH_SIZE = 16
RANDOM_SEED = 42
TRAIN_SPLIT_RATIO = 0.8
NUM_WORKERS = 0

def get_device():
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

def get_predictions_and_labels(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[List[float], List[float]]:
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Getting Predictions on {device}", leave=False)
        for images, labels, masks_from_collate in pbar: # masks_from_collate is float
            if images.numel() == 0: continue
            images = images.to(device)
            labels_cpu = labels.cpu().numpy()

            masks_float = masks_from_collate.to(device, dtype=torch.float32) 
            masks_bool = masks_from_collate.to(device, dtype=torch.bool)  

            outputs = model(images, outputs_mask=masks_float) 

            for i in range(outputs.shape[0]):
                mask_indices = torch.where(masks_bool[i, :, 0])[0] 
                if len(mask_indices) == 1:
                    valid_index = mask_indices[0]
                    pred = outputs[i, valid_index, 0].item()
                    label = labels_cpu[i, 0].item()
                    all_preds.append(pred)
                    all_labels.append(label)
                elif len(mask_indices) > 1:
                    print(f"Warning: Multiple valid mask indices found for sample {i} in a batch. Using the first one.")
                    valid_index = mask_indices[0]
                    pred = outputs[i, valid_index, 0].item()
                    label = labels_cpu[i, 0].item()
                    all_preds.append(pred)
                    all_labels.append(label)
    return all_preds, all_labels

def compute_aggregated_r2_for_dataset(
        model_paths: List[str],
        loader: DataLoader,
        device: torch.device,
        dataset_name: str,
        model_architecture_params: Dict): 
    """
    Computes aggregated R^2 from multiple models for a given DataLoader.
    """
    print(f"\n--- Processing for Aggregated R^2 on {dataset_name} Set ---")
    if not loader or len(loader.dataset) == 0:
        print(f"Skipping {dataset_name} set as it is empty or loader not provided.")
        return

    all_model_predictions: List[List[float]] = []
    true_labels: Optional[List[float]] = None

    for i, model_path in enumerate(model_paths):
        print(f"  Processing Model {i+1}/{len(model_paths)}: {os.path.basename(model_path)}")
        model = PyTorchResNetTransformer.get_model_with_head(model_architecture_params)
        
        if not os.path.exists(model_path):
            print(f"  Error: Model file not found at {model_path}. Skipping this model.")
            all_model_predictions.append([])
            continue
        try:
            model.to(device)
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
                model.load_state_dict(model_state_dict)
            elif isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"  Error loading model state_dict for {os.path.basename(model_path)}: {e}. Skipping this model.")
            all_model_predictions.append([])
            continue

        preds, current_labels = get_predictions_and_labels(model, loader, device)
        
        if not preds:
            print(f"  Warning: No predictions obtained for model {os.path.basename(model_path)} on {dataset_name} set. Skipping this model.")
            all_model_predictions.append([])
            continue

        all_model_predictions.append(preds)

        if true_labels is None:
            true_labels = current_labels
        elif len(true_labels) != len(current_labels) or not np.allclose(true_labels, current_labels, equal_nan=True):
            print(f"  Error: Label inconsistency between model runs on {dataset_name} set. This should not happen with shuffle=False.")
            return
        
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    valid_model_predictions = [p for p in all_model_predictions if p]

    if not valid_model_predictions:
        print(f"No valid predictions obtained from any model for {dataset_name} set. Cannot compute aggregated R^2.")
        return
    
    if true_labels is None or not true_labels:
        print(f"True labels were not collected for {dataset_name} set. Cannot compute R^2.")
        return

    num_samples_labels = len(true_labels)
    for i, preds in enumerate(valid_model_predictions):
        if len(preds) != num_samples_labels:
            print(f"  Error: Predictions list length ({len(preds)}) from a model "
                  f"does not match true labels length ({num_samples_labels}) for {dataset_name} set.")
            return

    try:
        predictions_array = np.array(valid_model_predictions)
    except ValueError as e:
        print(f"  Error converting predictions to NumPy array for {dataset_name} set: {e}")
        return

    if predictions_array.ndim == 1 and len(valid_model_predictions) == 1:
         aggregated_preds = predictions_array
    elif predictions_array.shape[0] != len(valid_model_predictions) or \
         (predictions_array.ndim > 1 and predictions_array.shape[1] != num_samples_labels) or \
         (predictions_array.ndim == 1 and len(predictions_array) != num_samples_labels and len(valid_model_predictions) > 1) :
        print(f"  Error: Prediction array shape is inconsistent for {dataset_name} set.")
        return
    else:
        aggregated_preds = np.mean(predictions_array, axis=0)

    if len(aggregated_preds) != len(true_labels):
        print(f"  Error: Aggregated predictions length ({len(aggregated_preds)}) "
              f"does not match true labels length ({len(true_labels)}) for {dataset_name} set.")
        return

    if len(aggregated_preds) < 2:
        print(f"  Not enough samples ({len(aggregated_preds)}) to compute a meaningful R^2 score for {dataset_name} set.")
        return

    finite_indices = np.isfinite(aggregated_preds) & np.isfinite(true_labels)
    if not np.all(finite_indices):
        print(f"  Warning: Found {np.sum(~finite_indices)} NaN/Inf values in aggregated predictions or labels. They will be removed before R2 calculation.")
        aggregated_preds_filtered = np.array(aggregated_preds)[finite_indices]
        true_labels_filtered = np.array(true_labels)[finite_indices]
    else:
        aggregated_preds_filtered = aggregated_preds
        true_labels_filtered = true_labels
    
    if len(aggregated_preds_filtered) < 2:
        print(f"  Not enough finite samples ({len(aggregated_preds_filtered)}) remaining after filtering to compute R^2 for {dataset_name} set.")
        return

    aggregated_r2 = r2_score(true_labels_filtered, aggregated_preds_filtered)
    print(f"Aggregated R^2 Score on {dataset_name} Set (from {len(valid_model_predictions)} models): {aggregated_r2:.4f}")


def main(model_paths: List[str]):
    print("--- Main Setup ---")
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
        print(f"Dataset loaded. Total samples: {len(full_dataset)}.")
    except FileNotFoundError as e:
        print(f"Fatal Error: Dataset file/directory not found: {e}")
        return
    except Exception as e:
         print(f"Fatal Error: An unexpected error occurred loading dataset: {e}")
         return

    if len(full_dataset) == 0:
        print("Fatal Error: Dataset is empty after loading. Exiting.")
        return

    print("Splitting dataset...")
    total_size = len(full_dataset)
    train_size = int(TRAIN_SPLIT_RATIO * total_size)
    val_size = total_size - train_size

    if train_size == 0 and val_size == 0 :
        print("Fatal Error: Both train and validation split sizes are zero. Exiting.")
        return

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_loader = None
    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=True if device.type == 'cuda' else False
        )
    else:
        print("Training dataset is empty. Skipping R^2 calculation for train set.")

    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=True if device.type == 'cuda' else False
        )
    else:
        print("Validation dataset is empty. Skipping R^2 calculation for validation set.")

    if train_loader:
        compute_aggregated_r2_for_dataset(
            model_paths=model_paths,
            loader=train_loader,
            device=device,
            dataset_name="Training",
            model_architecture_params=MODEL_PARAMS 
        )
    
    if val_loader:
        compute_aggregated_r2_for_dataset(
            model_paths=model_paths,
            loader=val_loader,
            device=device,
            dataset_name="Validation",
            model_architecture_params=MODEL_PARAMS
        )

    print("\n--- Evaluation Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple trained ResNet-Transformer models " 
                                                 "by averaging their predictions and computing a single R^2 "
                                                 "for both training and validation sets.")
    parser.add_argument("model_paths", type=str, nargs='+',
                        help="Paths to the trained model (.pth files).")
    args = parser.parse_args()

    if not args.model_paths:
        print("Error: No model paths provided.")
        exit()
    
    if len(args.model_paths) != 3:
        print(f"Warning: Expected 3 model paths for aggregation, but received {len(args.model_paths)}. Proceeding with the provided models.")

    if 'TFRecordSenegalDataset' in globals() and TFRecordSenegalDataset.__module__ != '__main__':
        if not os.path.isdir(TFRECORD_DIR):
            print(f"Error: TFRecord directory not found: {os.path.abspath(TFRECORD_DIR)}")
            exit()
        if not os.path.isfile(CSV_PATH):
            print(f"Error: CSV file not found: {os.path.abspath(CSV_PATH)}")
            exit()

    main(args.model_paths)