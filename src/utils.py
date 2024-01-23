import os
import shutil
import numpy as np
from tqdm import tqdm

def create_splits(data_dir: str = 'data/processed',
                  target_dir: str = 'data/splits',
                  dataset_size: int | None = None,
                  train_prop: float = 0.8, 
                  val_prop: float = 0.1, 
                  test_prop: float = 0.1) -> bool:
    """
    Create train, val, and test splits of data in data_dir.
    
    Parameters:
    - data_dir: str, path to data directory
    - target_dir: str, path to directory to save splits
    - dataset_size: int, number of images to use for creating splits, if None, use all images
    - train_prop: float, proportion of data to use for training
    - val_prop: float, proportion of data to use for validation
    - test_prop: float, proportion of data to use for testing
    
    Returns:
    - success: bool, whether splits were created successfully
    """
    # Get all files in data_dir
    data_dir = os.path.abspath(data_dir)
    files = os.listdir(data_dir)
    print(f'Found {len(files)} files in {data_dir}')
    
    # Create train, val, and test directories
    target_dir = os.path.abspath(target_dir)
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')
    
    # Shuffle files
    np.random.shuffle(files)
    
    # Use only a subset of the data if specified
    if dataset_size is not None:
        files = files[:dataset_size]
        print(f'Using {len(files)} images for creating splits')
    
    # Create train, val, and test splits
    train_cutoff = int(train_prop * len(files))
    val_cutoff = int((train_prop + val_prop) * len(files))
    
    train_files = files[:train_cutoff]
    val_files = files[train_cutoff:val_cutoff]
    test_files = files[val_cutoff:]
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
        
    # Copy files to train, val, and test directories
    print(f'Copying files from {data_dir} to {train_dir}, {val_dir}, and {test_dir}')
    for file in tqdm(train_files, total=len(train_files), desc='Copying train files'):
        shutil.copyfile(os.path.join(data_dir, file), os.path.join(train_dir, file))
    for file in tqdm(val_files, total=len(val_files), desc='Copying val files'):
        shutil.copyfile(os.path.join(data_dir, file), os.path.join(val_dir, file))
    for file in tqdm(test_files, total=len(test_files), desc='Copying test files'):
        shutil.copyfile(os.path.join(data_dir, file), os.path.join(test_dir, file))

        
    return True
