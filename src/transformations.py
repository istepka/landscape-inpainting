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
    
    def fit(self, X: torch.Tensor | np.ndarray):
        '''
        X: torch.Tensor of shape (N, W, H, C)
        '''
        assert len(X.shape) == 4, f'Expected X to have 4 dimensions, got {len(X.shape)}'
        assert X.shape[3] == 3, f'Expected X to have 3 channels, got {X.shape[1]}'

        epsilon = 1e-8
        self.mean = np.mean(X, axis=(0, 1, 2)) + epsilon
        self.std = np.std(X, axis=(0, 1, 2)) + epsilon
        
    def transform(self, X: torch.Tensor | np.ndarray):
        '''
        X: torch.Tensor of shape (N, W, H, C)
        '''
        assert len(X.shape) == 4, f'Expected X to have 4 dimensions, got {len(X.shape)}'
        assert X.shape[3] == 3, f'Expected X to have 3 channels, got {X.shape[1]}'
        
        
        after =  (X- self.mean) / self.std
        return after
    
    def inverse_transform(self, X: torch.Tensor):
        '''
        X: torch.Tensor of shape (N, W, H, C)
        '''
        assert len(X.shape) == 4, f'Expected X to have 4 dimensions, got {len(X.shape)}'
        assert X.shape[3] == 3, f'Expected X to have 3 channels, got {X.shape[1]}'
        
        after =  (X * self.std) + self.mean
        
        return after
    

    