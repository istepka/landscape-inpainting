import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import wandb
from PIL import Image

def create_splits(data_dir: str = 'data/processed',
                  target_dir: str = 'data/splits',
                  dataset_size: int | None = None,
                  train_size: int = 10_000,
                  val_size: int = 2_000,
                  test_size: int = 10_000 
    ) -> bool:
    """
    Create train, val, and test splits of data in data_dir.
    
    Parameters:
    - data_dir: str, path to data directory
    - target_dir: str, path to directory to save splits
    - dataset_size: int, number of images to use for creating splits, if None, use all images
    - train_size: int, number of images to use for training
    - val_size: int, number of images to use for validation
    - test_size: int, number of images to use for testing
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
    train_cutoff = train_size
    val_cutoff = train_size + val_size
    
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

def show_images(images: torch.Tensor,
                num_images: int=5, 
                wandb_save_name: str = None, 
                show_plot_locally: bool = True,
                base_dir: str = 'figures/',
    ):
    fig, ax = plt.subplots(1, num_images, figsize=(12, 3))
    for i in range(num_images):
        # print(images[i].shape, images[i].max(), images[i].min())
        ax[i].imshow(images[i].permute(1, 2, 0))
        ax[i].axis('off')
    plt.tight_layout()
    
    if wandb_save_name:
        save_path = os.path.join(base_dir, wandb_save_name)
        plt.savefig(save_path, dpi=300)
        wandb.log({wandb_save_name: wandb.Image(save_path)})
    if show_plot_locally:
        plt.show()

# Create a couple of shapes that will be used to randomly paste onto the images
def create_circle(size: int) -> np.array:
    circle = np.zeros((size, size))
    x, y = np.indices((size, size))
    circle = np.logical_or(circle, (x - size//2)**2 + (y - size//2)**2 <= (size//2)**2)
    circle3D = np.zeros((size, size, 3))
    circle3D[:, :, 0] = circle
    circle3D[:, :, 1] = circle
    circle3D[:, :, 2] = circle
    return circle3D.astype(np.uint8)

def create_square(size: int) -> np.array:
    square = np.zeros((size, size))
    square = np.logical_or(square, np.ones((size, size)))
    square3D = np.zeros((size, size, 3))
    square3D[:, :, 0] = square
    square3D[:, :, 1] = square
    square3D[:, :, 2] = square
    return square3D.astype(np.uint8)

# Create a function that will randomly paste shapes onto the images
def paste_shape(image: np.array, shape: np.array) -> np.array:
    assert image.shape[2] == shape.shape[2], \
        f'Image and shape must have the same channel dimensions but got {image.shape} and {shape.shape}'
    
    # Randomly choose a location to paste the shape
    x = np.random.randint(0, image.shape[0] - shape.shape[0])
    y = np.random.randint(0, image.shape[1] - shape.shape[1])
    
    # Create a mask of the shape
    mask = shape == 1
    
    # Paste the shape onto the image
    image[x:x+shape.shape[0], y:y+shape.shape[1], :] = np.where(mask, shape, image[x:x+shape.shape[0], y:y+shape.shape[1], :])
    
    return image

def load_images_from_dir(dir: str) -> np.array:
    images = []
    for filename in os.listdir(dir):
        if any([filename.lower().endswith(x) for x in ['.jpg', '.png', '.jpeg']]):
            
            try:
                image = Image.open(os.path.join(dir, filename))
                images.append(np.array(image))
            except Exception as e:
                print(f'Skipping one image due to loading error {filename}: {e}')
    return np.array(images)



