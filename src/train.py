import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning import LightningModule
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import mlflow
import os
import sys
import warnings

if __name__ == '__main__':
    # Seed for reproducibility
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    L.seed_everything(seed)

    # --- PREP ENVIRONMENT --- #

    # Set up all directories
    data_dir = 'data'
    model_dir = 'models'
    figure_dir = 'figures'
    mlflow_dir = 'mlflow'

    # Create directories if they don't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    if not os.path.exists(mlflow_dir):
        os.makedirs(mlflow_dir)
        
    # Set up MLFlow
    mlflow.set_tracking_uri('file:' + os.path.abspath(mlflow_dir))
    mlflow.set_experiment('CV_Project3')