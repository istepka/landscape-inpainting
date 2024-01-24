from PIL import Image

# Remove images from processed folder that don't have 3 channels

if __name__ == '__main__':
    import os
    import sys
    import shutil
    import numpy as np

    # Load the data
    data_dir = 'data/processed'
    data_paths = []
    for f in os.listdir(data_dir):
        if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']:
            data_paths.append(os.path.join(data_dir, f))
            
    print(f'Loaded {len(data_paths)} image paths into memory')
    
    # Remove images that don't have 3 channels
    for i, path in enumerate(data_paths):
        img = Image.open(path)
        img = np.array(img)
        if len(img.shape) != 3:
            print(f'Removing {path}')
            os.remove(path)
            
    # Remove empty folders
    for folder in os.listdir(data_dir):
        if len(os.listdir(os.path.join(data_dir, folder))) == 0:
            print(f'Removing empty folder {folder}')
            os.rmdir(os.path.join(data_dir, folder))
            
    print('Done!')