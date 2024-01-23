import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformations import TransformClass
import lightning as L

class CustomDataset(Dataset):
    def __init__(self, name: str, data_dir: str, pretransform: TransformClass, transform: TransformClass = None, load_into_memory=True, device='cuda'):
        '''
        Parameters:
        - name: str, the name of the dataset (train, val, test)
        - data_dir: str, the directory where the data is stored
        - pretransform: TransformClass, the pretransform to apply to the data in bulk before using it e.g. normalization
        - transform: TransformClass, the transform to apply to the data in real time like augmentation
        - load_into_memory: bool, whether to load the data into memory or not
        - device: str, the device to load the data onto 
        '''
        self.data_dir = data_dir
        self.transform = transform
        self.pretransform = pretransform
        self.data = self.load_data()
        self.load_into_memory = load_into_memory
        self.device = device
        self.name = name
        
    def load_data(self):
        self.data_paths = []
        for f in os.listdir(self.data_dir):
            if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']:
                self.data_paths.append(os.path.join(self.data_dir, f))
        
        if self.name == 'train':
            self.data = []
            for f in self.data_paths:
                try:
                    self.data.append(Image.open(f))
                except Exception as e:
                    print(f'Failed to load {f}, because {e}')
                    
            # Fit the Pretransform on the data
            print(f'Fitting pretransform on {len(self.data)} images')
            self.pretransform.fit(self.data)
            print(f'Performed fit on {len(self.data)} images')
            
            if not self.load_into_memory:    
                # remove the data from memory 
                del self.data 
                # replace with the data paths
                self.data = self.data_paths  
        else:
            if not self.load_into_memory:
                self.data = self.data_paths
                print(f'Loaded {len(self.data)} image paths into memory')
                
        if self.load_into_memory:
            self.data = torch.stack(self.data).to(self.device)
            # Reshape to (N, C, W, H)
            self.data = self.data.permute(0, 3, 1, 2)
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
            sample = torch.tensor(sample).to(self.device)
            sample = sample.permute(2, 0, 1)
            sample = sample.unsqueeze(0)
            sample = self.pretransform.transform(sample)
            sample = sample.squeeze(0)
        
        # Apply the transform in real time
        if self.transform:
            sample = self.transform(sample)
        return sample

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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True).to(self.device)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers).to(self.device)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers).to(self.device)
