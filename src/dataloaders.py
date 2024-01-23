import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformations import TransformClass
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
                 fit_pretransform_size: float = 0.1
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
        
        # Load the data
        self.data = self.load_data(fit_pretransform_size=fit_pretransform_size)
        
    def load_data(self, fit_pretransform_size: float = 0.1):
        self.data_paths = []
        for f in os.listdir(self.data_dir):
            if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']:
                self.data_paths.append(os.path.join(self.data_dir, f))
        
        if self.name == 'train':
            self.data = []
            for f in self.data_paths:
                try:
                    self.data.append(np.array(Image.open(f)))
                except Exception as e:
                    print(f'Failed to load {f}, because {e}')
                    
                if fit_pretransform_size * len(self.data_paths) < len(self.data):
                    print(f'Loaded {len(self.data)} images for pretransform fitting')
                    break
            
            self.data = np.array(self.data)
            self.data = torch.Tensor(self.data).permute(0, 3, 1, 2) # Reshape to (N, C, W, H)
            if self.device == 'cuda':
                self.data = self.data.to(self.device)
            # Fit the Pretransform on the data
            print(f'Fitting pretransform on {len(self.data)} images. Dims: {self.data.shape}')
            self.pretransform.fit(self.data)
            print(f'Performed fit on {len(self.data)} images')
            del self.data 
            if not self.load_into_memory:
                self.data = self.data_paths
                print(f'Loaded {len(self.data)} image paths into memory')
        else:
            if not self.load_into_memory:
                self.data = self.data_paths
                print(f'Loaded {len(self.data)} image paths into memory')
                
        if self.load_into_memory:
            self.data = []
            for f in self.data_paths:
                try:
                    a = np.array(Image.open(f))
                    assert a.shape[2] == 3, f'Image {f} has {a.shape[2]} channels, but should have 3'
                    self.data.append(a)
                except Exception as e:
                    print(f'Failed to load {f}, because {e}')
                    
            self.data = np.array(self.data)
            self.data = torch.Tensor(self.data).permute(0, 3, 1, 2) # Reshape to (N, C, W, H)
            if self.device == 'cuda':
                self.data = self.data.to(self.device)
            print(f'Loaded {len(self.data)} images into memory: their shape is {self.data.shape}')
            
            # Perform the pretransform
            print(f'Pretransforming {len(self.data)} images')
            self.data = self.pretransform.transform(self.data)
            print(f'Performed pretransform on {len(self.data)} images')
                
        return self.data
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # If we're not loading into memory, load the image
        if not self.load_into_memory:
            sample = Image.open(sample)
            sample = np.array(sample)
            sample = torch.tensor(sample)
            sample = sample.to(self.device)
            sample = sample.permute(2, 0, 1)
            sample = sample.unsqueeze(0)
            sample = self.pretransform.transform(sample)
            sample = sample.squeeze(0)
        
        # Apply the transform in real time
        if self.transform:
            sample = self.transform(sample)
        
        mask = torch.zeros_like(sample)

        mask[:, mask.shape[1] // 4: 3 * mask.shape[1] // 4, mask.shape[2] // 4: 3 * mask.shape[2] // 4] = 0
        mask = mask.to(self.device)

        return sample * mask, mask


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
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomDataset('train', os.path.join(self.data_dir, 'train'), pretransform=self.pretransform, transform=self.transform, load_into_memory=self.load_into_memory, device=self.device)
            self.val_dataset = CustomDataset('val', os.path.join(self.data_dir, 'val'), pretransform=self.pretransform, transform=self.transform, load_into_memory=self.load_into_memory, device=self.device)
        if stage == 'test' or stage is None:
            self.test_dataset = CustomDataset('test', os.path.join(self.data_dir, 'test'), pretransform=self.pretransform, transform=self.transform, load_into_memory=self.load_into_memory, device=self.device)

    def train_dataloader(self):
        dl = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return dl