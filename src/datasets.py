import numpy as np
import torch
from .utils import paste_shape

class ShapesDataset(torch.utils.data.Dataset):
    def __init__(self, images: np.array, masks: list, transform: object = None, fill_color: str = 'black'):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.fill_color = fill_color
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].copy()
        mask = np.zeros_like(image)
        
        # Randomly choose a number of shapes to paste onto the image
        num_shapes = np.random.randint(1, 10)
        
        ms = [self.masks[i] for i in np.random.randint(0, len(self.masks), num_shapes)]
        masked_image = image.copy()
        
        for m in ms:
            mask = paste_shape(mask, m)
        
        if self.fill_color == 'black':
            masked_image = masked_image * (1-mask)
        else:
            masked_image = masked_image * mask
            
        sample = {'image': image, 'mask': mask, 'masked_image': masked_image.copy()}
        
        if self.transform:
            sample = self.transform(sample)
        return sample
    
# Create a transform class to convert the images to tensors
class ToTensor(object):
    def __call__(self, sample):
        image, mask, masked_image = sample['image'], sample['mask'], sample['masked_image']
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        masked_image = masked_image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'mask': torch.from_numpy(mask).float(),
                'masked_image': torch.from_numpy(masked_image).float()}