import numpy as np 
from PIL import Image
import os
import multiprocessing as mp
import argparse
import shutil

def preprocess_image(img_file: str, size: tuple = (256, 256)) -> np.ndarray:
    """
    Preprocess an image file into a numpy array of shape (W, H, 3).
    
    Parameters:
    - img_file: str, path to image file
    - size: tuple, desired size of image
    
    Returns:
    - img: np.ndarray, preprocessed image of shape (N, W, H, 3)
    
    """
    w = size[0]
    h = size[1]
    
    # If image is more than 2x larger than desired size in both dimensions,
    # We want to extract all non-overlapping crops of size (w, h) from the image
    # Otherwise, we want to resize the image to (w, h)
    img = Image.open(img_file)
    img_w, img_h = img.size
    
    if img_w >= 2 * w and img_h >= 2 * h:
        # Extract crops
        crops = []
        for i in range(0, img_w - w, w):
            for j in range(0, img_h - h, h):
                crop = img.crop((i, j, i + w, j + h))
                crops.append(np.array(crop))
                
        return np.array(crops)
    else:
        # Resize image
        img = img.resize((w, h))
        return np.array(img).reshape(1, w, h, 3)

def preprocess_data(data_dir: str, 
                    out_dir: str, 
                    size: tuple = (256, 256), 
                    num_workers: int = mp.cpu_count()) -> bool:
    """
    Preprocess all images in data_dir and save to out_dir.
    
    Parameters:
    - data_dir: str, path to raw data directory
    - out_dir: str, path to save processed data
    - size: tuple, desired size of images
    - num_workers: int, number of workers to use for multiprocessing
    
    Returns:
    - success: bool, whether preprocessing was successful
    """
    
    # Find all image files in data_dir
    ext = ['.jpg', '.jpeg', '.png']
    img_files = []
    
    def __recursive_find(dir: str) -> None:
        for f in os.listdir(dir):
            if os.path.isdir(os.path.join(dir, f)):
                __recursive_find(os.path.join(dir, f))
            elif os.path.splitext(f)[1].lower() in ext:
                img_files.append(os.path.join(dir, f))
                
    __recursive_find(data_dir)
    
    print(f'Found {len(img_files)} images in {data_dir}')
    
    # Preprocess in parallel
    processes = []
    img_files_split = np.array_split(img_files, num_workers)
    
   
    
    try:
        # Remove out_dir if it already exists
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
            
        # Create out_dir
        os.makedirs(out_dir, exist_ok=True)
        
        # Start processes
        for i in range(num_workers):
            p = mp.Process(target=__preprocess_worker, args=(img_files_split[i], out_dir, size))
            processes.append(p)
            p.start()
            
        for p in processes:
            p.join()
            
        print(f'Preprocessed images saved to {out_dir}')
        
        return True
    except Exception as e:
        print(e)
        return False

def __preprocess_worker(img_files: list, out_dir: str, size: tuple) -> None:
    _failed = 0
    _saved = 0
    for i, img_file in enumerate(img_files):
        try:
            img = preprocess_image(img_file, size)
            
            # Check if has 3 channels
            if len(img.shape) != 4:
                print(f'Image {img_file} has {len(img.shape)} channels, skipping')
                _failed += 1
                continue
            
            for j, im in enumerate(img):
                im = Image.fromarray(im)
                
                img_name = "".join([c for c in np.random.choice(list('abcdefghijklmnopqrstuvwxyz1234567890'), 25)]) + '.jpg'
                img_dir = os.path.join(out_dir, img_name)
                
                im.save(img_dir)
                _saved += 1
                
        except Exception as e:
            print(e)
            print(f'Failed to preprocess image {img_file}')
            _failed += 1
            
    print(f'Worker {mp.current_process().name} finished preprocessing {len(img_files)} images. \
        \nFailed to preprocess {_failed} images out of {len(img_files)}' \
        f'\nSaved {_saved} images to {out_dir}'
        )

        
if __name__ == '__main__':
    
    arparser = argparse.ArgumentParser()
    arparser.add_argument('--data_dir', type=str, default='data/raw', help='Path to raw data directory')
    arparser.add_argument('--out_dir', type=str, default='data/processed', help='Path to save processed data')
    arparser.add_argument('--size', type=int, default=256, help='Desired size of images (w=h) so just one number')
    arparser.add_argument('--num_workers', type=int, default=mp.cpu_count(), help='Number of workers to use for multiprocessing')
    args = arparser.parse_args()
    
    success = preprocess_data(args.data_dir, args.out_dir, (args.size, args.size), args.num_workers)
    
    if success:
        print('Preprocessing successful!')
    else:
        print('Preprocessing failed!')


    # Verify how many images we have in out_dir
    print(len(os.listdir(args.out_dir)))
    