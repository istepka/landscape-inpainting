import os
import shutil
import numpy as np

def create_splits(data_dir: str = 'data/processed',
                  target_dir: str = 'data/splits',
                  train_prop: float = 0.8, 
                  val_prop: float = 0.1, 
                  test_prop: float = 0.1) -> bool:
    """
    Create train, val, and test splits of data in data_dir.
    
    Parameters:
    - data_dir: str, path to data directory
    - target_dir: str, path to directory to save splits
    - train_prop: float, proportion of data to use for training
    - val_prop: float, proportion of data to use for validation
    - test_prop: float, proportion of data to use for testing
    
    Returns:
    - success: bool, whether splits were created successfully
    """
    # Get all files in data_dir
    files = os.listdir(data_dir)
    
    # Create train, val, and test directories
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')
    
    # Shuffle files
    np.random.shuffle(files)
    
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
    for file in train_files:
        shutil.copy(os.path.join(data_dir, file), os.path.join(train_dir, file))
    for file in val_files:
        shutil.copy(os.path.join(data_dir, file), os.path.join(val_dir, file))
    for file in test_files:
        shutil.copy(os.path.join(data_dir, file), os.path.join(test_dir, file))
        
    return True
