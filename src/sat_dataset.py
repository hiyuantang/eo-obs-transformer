import os
import re
import torch
import pandas as pd
import tensorflow as tf
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
import warnings

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', message=r'.*tf.io.gfile.Glob.*')

class TFRecordSenegalDataset(Dataset):
    """
    PyTorch Dataset for loading Senegal TFRecord time series data,
    where each record contains multiple features with band/year in the key
    and float_list image data.

    Args:
        tfrecord_dir (str): Path to the base directory containing subdirectories
                            with .tfrecord.gz files.
        csv_path (str): Path to the senegal_shuffled.csv file.
        target_bands (list[str]): Desired order of bands in the output tensor.
        image_size (int): The expected height and width of the images (e.g., 224).
    """
    def __init__(self, tfrecord_dir, csv_path, target_bands, image_size=224): 
        super().__init__()
        self.tfrecord_dir = tfrecord_dir
        self.csv_path = csv_path
        self.target_bands = target_bands
        self.band_to_index = {band_name: i for i, band_name in enumerate(target_bands)}
        self.num_bands = len(target_bands)
        self.image_height = image_size
        self.image_width = image_size
        self.expected_pixels = self.image_height * self.image_width

        # --- CSV Loading (same as before) ---
        if not os.path.isdir(self.tfrecord_dir):
             raise FileNotFoundError(f"TFRecord directory not found: {self.tfrecord_dir}")
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        print(f"Loading CSV data from: {self.csv_path}")
        self.metadata_df = pd.read_csv(self.csv_path)
        self.metadata_lookup = {}
        for year, group in self.metadata_df.groupby('year'):
            self.metadata_lookup[year] = group.reset_index(drop=True)['iwi'].to_dict()
        # --- End CSV Loading ---

        print(f"Scanning for TFRecord files recursively in: {self.tfrecord_dir}")
        self.tfrecord_files = []
        filename_pattern = re.compile(r"([^_]+)_(\d{4})_(\d+)\.tfrecord\.gz$")

        for root, dirs, files in os.walk(self.tfrecord_dir):
            for filename in files:
                match = filename_pattern.search(filename)
                if match and filename.endswith(".tfrecord.gz"):
                    country, year_str, index_str = match.groups()
                    year = int(year_str)
                    relative_index = int(index_str)
                    full_path = os.path.join(root, filename)

                    try:
                        if year in self.metadata_lookup and relative_index in self.metadata_lookup[year]:
                             iwi_label = self.metadata_lookup[year][relative_index]
                             self.tfrecord_files.append({
                                 "path": full_path,
                                 "country": country,
                                 "target_year": year,
                                 "relative_index": relative_index,
                                 "iwi": float(iwi_label)
                             })
                    except Exception as e:
                        print(f"Warning: Error processing metadata for {filename}. Error: {e}. Skipping.")

        if not self.tfrecord_files:
             raise FileNotFoundError(f"No matching .tfrecord.gz files found recursively in {self.tfrecord_dir}")
        print(f"Found {len(self.tfrecord_files)} TFRecord files.")

        self.key_parser_regex = re.compile(r'_([A-Z0-9]+)_(\d{4})_(\d{4})$')


    def __len__(self):
        return len(self.tfrecord_files)

    def __getitem__(self, idx):
        file_info = self.tfrecord_files[idx]
        tfrecord_path = file_info["path"]
        target_year = file_info["target_year"]
        iwi_label = file_info["iwi"]

        images_by_year_band = defaultdict(dict)

        try:
            # Read the *first* (and likely only) record from the TFRecord file
            raw_dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='GZIP').take(1)
            example_proto = list(raw_dataset)[0] # Get the single raw record
            example = tf.train.Example()
            example.ParseFromString(example_proto.numpy())

            # Iterate through all features found in this single record
            for key, feature in example.features.feature.items():
                if feature.WhichOneof('kind') == 'float_list':
                    # Attempt to parse band and year from the feature key
                    match = self.key_parser_regex.search(key)
                    if match:
                        band_type = match.group(1)
                        band_year = int(match.group(2)) # Use start year

                        if band_type in self.band_to_index:
                            # Extract float list and check dimensions
                            pixel_values = feature.float_list.value
                            if len(pixel_values) == self.expected_pixels:
                                # Reshape flat list into HxW image
                                image_np = np.array(pixel_values, dtype=np.float32).reshape(
                                    (self.image_height, self.image_width)
                                )
                                # Store as torch tensor
                                images_by_year_band[band_year][band_type] = torch.from_numpy(image_np)
                            else:
                                print(f"Warning: Feature '{key}' in {tfrecord_path} has {len(pixel_values)} values, "
                                      f"expected {self.expected_pixels} ({self.image_height}x{self.image_width}). Skipping.")

        except IndexError:
             print(f"Error: TFRecord file {tfrecord_path} appears to be empty or corrupted. Skipping.")
             raise IOError(f"Failed to read record from {tfrecord_path}")
        except tf.errors.DataLossError as e:
             print(f"Error: Data loss error reading {tfrecord_path}. Skipping file. Details: {e}")
             raise IOError(f"Failed to read {tfrecord_path} due to DataLossError") from e
        except Exception as e:
             print(f"Error processing TFRecord {tfrecord_path}: {e}")
             import traceback
             traceback.print_exc()
             raise IOError(f"Failed to process {tfrecord_path}") from e

        final_images_by_year = {}
        processed_years = sorted(images_by_year_band.keys())

        for year in processed_years:
            bands_present = images_by_year_band[year]
            if len(bands_present) == self.num_bands:
                year_tensor = torch.zeros((self.num_bands, self.image_height, self.image_width), dtype=torch.float32)
                for band_type, band_tensor in bands_present.items():
                    if band_type in self.band_to_index: 
                         band_idx = self.band_to_index[band_type]
                         year_tensor[band_idx] = band_tensor

                final_images_by_year[year] = year_tensor


        output = {
            "images_by_year": final_images_by_year,
            "target_year": target_year,
            "iwi": torch.tensor(iwi_label, dtype=torch.float32)
        }

        return output

if __name__ == '__main__':
    TFRECORD_DIR = './sat_data'
    CSV_PATH = './senegal_shuffled.csv' 

    TARGET_BANDS = [
        'BLUE', 'GREEN', 'RED',
        'NIR', 'SWIR1', 'SWIR2',
        'TEMP1',
        'NIGHTLIGHTS'  
    ]
    IMAGE_SIZE = 224 

    if not os.path.exists(TFRECORD_DIR) or not os.path.exists(CSV_PATH):
         print("Error: TFRecord directory or CSV file not found at specified paths.")
         print(f"Checked TFRECORD_DIR: {os.path.abspath(TFRECORD_DIR)}")
         print(f"Checked CSV_PATH: {os.path.abspath(CSV_PATH)}")
         print("Please update TFRECORD_DIR and CSV_PATH variables in the script.")
    else:
        print("\n--- Initializing Dataset ---")
        try:
            senegal_dataset = TFRecordSenegalDataset(
                tfrecord_dir=TFRECORD_DIR,
                csv_path=CSV_PATH,
                target_bands=TARGET_BANDS,
                image_size=IMAGE_SIZE # Pass image size
            )
            print("\n--- Dataset Initialized ---")

            print(f"Total number of samples (locations): {len(senegal_dataset)}")

            if len(senegal_dataset) > 0:
                print("\n--- Getting first sample ---")
                try:
                    first_sample = senegal_dataset[0]
                    print("Sample loaded successfully!")
                    print("Keys:", first_sample.keys())
                    print("Target Year:", first_sample['target_year'])
                    print("IWI Label:", first_sample['iwi'])
                    print("Image data keys (years found):", sorted(list(first_sample['images_by_year'].keys())))

                    if first_sample['images_by_year']:
                        first_year_key = sorted(list(first_sample['images_by_year'].keys()))[0]
                        print(f"Shape of image tensor for year {first_year_key}:",
                              first_sample['images_by_year'][first_year_key].shape)
                        print(f"dtype: {first_sample['images_by_year'][first_year_key].dtype}")
                    else:
                        print("No valid image data found for this sample after processing.")

                except IndexError as e:
                     print(f"Error getting item 0: {e}. Is the dataset empty after filtering or scanning?")
                except IOError as e:
                    print(f"Error getting item 0: {e}. File reading or processing failed.")
                except Exception as e:
                     print(f"An unexpected error occurred getting item 0: {e}")
                     import traceback
                     traceback.print_exc()

            # --- DataLoader Test (same as before) ---
            print("\n--- Testing DataLoader (num_workers=0) ---")
            try:
                 dataloader = torch.utils.data.DataLoader(senegal_dataset, batch_size=4, shuffle=True, num_workers=0)
                 batch = next(iter(dataloader))
                 print("Batch loaded successfully!")
                 print("Batch keys:", batch.keys())
                 print("Batch target years:", batch['target_year'])
                 print("Batch IWI:", batch['iwi'])
            except Exception as e:
                 print(f"\nError creating or iterating DataLoader: {e}")

        except FileNotFoundError as e:
             print(f"Initialization failed: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during dataset initialization: {e}")
             import traceback
             traceback.print_exc()