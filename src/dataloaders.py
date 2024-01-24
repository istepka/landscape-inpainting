import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformations import TransformClass, TransformStandardScaler
import lightning as L
import numpy as np


class CustomDataset(Dataset):
    
    def __init__(self, 
                 name: str, 
                 data_dir: str, 
                 pretransform: TransformClass, 
                 transform: TransformClass = None, 
                 load_into_memory=True, 
                 device='cuda',
                 fit_pretransform_size: float = 1.0
                 ) -> None:
        '''
        Parameters:
        - name: str, the name of the dataset (train, val, test)
        - data_dir: str, the directory where the data is stored
        - pretransform: TransformClass, the pretransform to apply to the data in bulk before using it e.g. normalization
        - transform: TransformClass, the transform to apply to the data in real time like augmentation
        - load_into_memory: bool, whether to load the data into memory or not
        - device: str, the device to load the data onto 
        '''
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.pretransform = pretransform
        self.load_into_memory = load_into_memory
        self.device = device
        self.name = name
        self.__data = None
 
        # Load the data
        self.load_data()
        
    def load_data(self):
        self.data_paths = []
        for f in os.listdir(self.data_dir):
            if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']:
                self.data_paths.append(os.path.join(self.data_dir, f))
            
        self.__data = np.array(self.data_paths)
        print(f'Loaded {len(self.__data)} image paths into memory')
        

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        # path = self.__data[idx]
        
        # sample = Image.open(path)
        # sample = np.array(sample).astype(np.float32)
        # if self.pretransform:
        #     sample = sample.reshape(1, sample.shape[0], sample.shape[1], sample.shape[2])
        #     sample = self.pretransform.transform(sample)
        #     sample = sample.reshape(sample.shape[1], sample.shape[2], sample.shape[3])
        #     # print('a', sample.shape)
        
        # # Apply the transform in real time
        # if self.transform:
        #     sample = self.transform(sample)
        
        # mask = np.ones_like(sample)
        # mask[mask.shape[1] // 4: 3 * mask.shape[1] // 4, mask.shape[2] // 4: 3 * mask.shape[2] // 4, :] = 0
        # sample = torch.tensor(sample).to(self.device)
        # mask = torch.tensor(mask).to(self.device)

        # return sample.permute(2, 0, 1), mask.permute(2, 0, 1) # (C, W, H)
        return self.__getitems__([idx])
    
    def __getitems__(self, idxs):
        paths = self.__data[idxs]
        
        samples = np.array([np.array(Image.open(path)) for path in paths]).astype(np.float32)
        if self.pretransform:
            samples = self.pretransform.transform(samples)
            # samples = samples.reshape(samples.shape[0], samples.shape[3], samples.shape[1], samples.shape[2])
            # print(self.pretransform.mean, self.pretransform.std)
            # print(samples.mean(), samples.std())
            
        # Apply the transform in real time
        if self.transform:
            samples = self.transform.transform(samples)
            
        # print('np', samples.mean(), samples.std())
            
        w, h = samples.shape[1], samples.shape[2]
        mask = np.ones(shape=(w, h, 3))
        mask[w // 4: 3 * w // 4, h // 4: 3 * h // 4, :] = 0
        masks = np.array([mask for _ in range(samples.shape[0])])
        
        # masks[:, masks.shape[1] // 4: 3 * masks.shape[1] // 4, masks.shape[2] // 4: 3 * masks.shape[2] // 4, :] = 0
        samples = torch.tensor(samples)
        masks = torch.tensor(masks)
        # print(samples.shape, masks.shape)
        # print('tensor', samples.mean(), samples.std())
        
        # print(f'Smaples shape {samples.shape}, masks shape {masks.shape}')
        
        return samples.permute(0, 3, 1, 2), masks.permute(0, 3, 1, 2) # (N, C, W, H)

class CustomDataModule(L.LightningDataModule):
    def __init__(self, 
                 data_dir: str, 
                 pretransform: TransformClass, 
                 batch_size: int = 32, 
                 num_workers: int = 4,  
                 transform: TransformClass = None,
                 device: str = 'cuda',
                 load_into_memory: bool = True
                 ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.pretransform = pretransform
        self.device = device
        self.load_into_memory = load_into_memory

    def setup(self, stage=None):
        # Initialize the dataset for each stage (train, val, test)
        if stage == 'fit':
            self.train_dataset = CustomDataset('fit', os.path.join(self.data_dir, 'train'), pretransform=None, transform=self.transform, load_into_memory=self.load_into_memory, device=self.device)
        if stage == 'train' or stage is None:
            self.train_dataset = CustomDataset('train', os.path.join(self.data_dir, 'train'), pretransform=self.pretransform, transform=self.transform, load_into_memory=self.load_into_memory, device=self.device)
            self.val_dataset = CustomDataset('val', os.path.join(self.data_dir, 'val'), pretransform=self.pretransform, transform=self.transform, load_into_memory=self.load_into_memory, device=self.device)
        if stage == 'test' or stage is None:
            self.test_dataset = CustomDataset('test', os.path.join(self.data_dir, 'test'), pretransform=self.pretransform, transform=self.transform, load_into_memory=self.load_into_memory, device=self.device)

    def train_dataloader(self):
        dl = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return dl

    def val_dataloader(self):
        dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return dl
    
    def test_dataloader(self):
        dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return dl