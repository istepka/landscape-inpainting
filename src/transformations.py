import numpy as np
import torch 

class TransformClass:
    def __init__(self):
        pass
    
    def fit(self, X):
        pass
    
    def transform(self, X):
        pass
    
    def inverse_transform(self, X):
        pass

class TransformStandardScaler(TransformClass):
    def __init__(self):
        self.mean = None
        self.std = None
        self.max = None
        self.min = None
    
    def fit(self, X: torch.Tensor):
        '''
        X: torch.Tensor of shape (N, C, W, H)
        '''
        assert len(X.shape) == 4, f'Expected X to have 4 dimensions, got {len(X.shape)}'
        assert X.shape[1] == 3, f'Expected X to have 3 channels, got {X.shape[1]}'
        
        self.mean = torch.mean(X, dim=(0, 2, 3))
        self.std = torch.std(X, dim=(0, 2, 3))
        # self.max = torch.max(X, dim=(0, 2, 3))
        # self.min = torch.min(X, dim=(0, 2, 3))
        
    def transform(self, X: torch.Tensor):
        '''
        X: torch.Tensor of shape (N, C, W, H)
        '''
        assert len(X.shape) == 4, f'Expected X to have 4 dimensions, got {len(X.shape)}'
        assert X.shape[1] == 3, f'Expected X to have 3 channels, got {X.shape[1]}'
        
        return ((X.permute(0, 2, 3, 1) - self.mean) / self.std).permute(0, 3, 1, 2)
    
    def inverse_transform(self, X: torch.Tensor):
        '''
        X: torch.Tensor of shape (N, C, W, H)
        '''
        assert len(X.shape) == 4, f'Expected X to have 4 dimensions, got {len(X.shape)}'
        assert X.shape[1] == 3, f'Expected X to have 3 channels, got {X.shape[1]}'
  
        return  ((X.permute(0, 2, 3, 1) * self.std) + self.mean).permute(0, 3, 1, 2) #TODO: find a better way to do this
    
    
    
    