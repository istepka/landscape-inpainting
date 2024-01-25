import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L 
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import wandb
import argparse
import time

from src.utils import *
from src.datasets import *
from src.models import *

config={
    "learning_rate": 1e-3,
    "epochs": 20,
    "batch_size": 32,
    "fill": 'black', # Can be also white
    "model": 'UNet', # Can be also UNet
    "loss": 'mse',
    "dataset_size": 22_000,
    "train_size": 10_000,
    "val_size": 2_000,
    "test_size": 10_000,
    'optimizer': 'adam',
    'experiment_name': 'default'
}


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--loss', type=str, choices=['mse', 'l1', 'cross_entropy', 'poisson', 'kldiv'])
parser.add_argument('--model', type=str, choices=['Encoder-Decoder', 'UNet'])
parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd', 'rmsprop', 'adagrad'])
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--experiment_name', type=str)

args = parser.parse_args()

for k, v in args.__dict__.items():
    if v is not None:
        config[k] = v



if not os.path.exists('data/splits'):
    create_splits(
        data_dir='data/processed', 
        target_dir='data/splits', 
        dataset_size=config['dataset_size'],
        train_size=config['train_size'],
        val_size=config['val_size'],
        test_size=config['test_size'],
    )

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

L.seed_everything(42)
np.random.seed(42)


# ----------
# WANDB
# ----------

with open('/workspace/wandb.key', 'r') as f:
    wandb_key = f.read().strip()
    
print('Logging in to wandb...')
wandb.login(key=wandb_key)
wandb.init(
    project='lightning', 
    name='UNet-largesample' if config['experiment_name'] is None else config['experiment_name'],
    )

wandb_logger = L.loggers.WandbLogger(
    project='lightning', 
    name='UNet-largesample' if config['experiment_name'] is None else config['experiment_name'],
)


DATA_DIRECTORY = '/workspace/data/splits/'
train_dir = os.path.join(DATA_DIRECTORY, 'train')
val_dir = os.path.join(DATA_DIRECTORY, 'val')

# Load images
train_images = load_images_from_dir(train_dir) / 255.0
val_images = load_images_from_dir(val_dir) / 255.0

print(f'Loaded {len(train_images)} training images')
print(f'Loaded {len(val_images)} validation images')


# Log sample images to wandb
fig, ax = plt.subplots(3, 5, figsize=(12, 7))
ax = ax.flatten()
for i in range(15):
    ax[i].imshow(train_images[i])
    ax[i].axis('off')
plt.tight_layout()
plt.savefig('train_images.png', dpi=300)
wandb.log({"train_images_sample": wandb.Image('train_images.png')})


# Create masks
circle_small = create_circle(32)
circle_large = create_circle(64)
square = create_square(32)
square_large = create_square(64)

masks = [circle_small, circle_large, square, square_large]
mask_names = ['circle_small', 'circle_large', 'square', 'square_large']

# Show the shapes
fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    ax[i].imshow(255 - masks[i] * 255)
    ax[i].set_title(mask_names[i])
    
plt.tight_layout()
plt.savefig('mask_shapes.png', dpi=300)
wandb.log({"mask_shapes": wandb.Image('mask_shapes.png')})


# Example of applying the masks to an image
fig, ax = plt.subplots(3, 5, figsize=(12, 7))
ax = ax.flatten()
for i in range(15):
    image = train_images[i].copy()
    for mask in masks:
        image = paste_shape(image, mask)
    ax[i].imshow(image)
    ax[i].axis('off')
plt.tight_layout()
plt.savefig('train_images_with_shapes.png', dpi=300)
wandb.log({"train_images_with_shapes": wandb.Image('train_images_with_shapes.png')})


# Create the datasets
train_dataset = ShapesDataset(train_images, masks, transform=ToTensor())
val_dataset = ShapesDataset(val_images, masks, transform=ToTensor())

# Create the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

# Show a batch of images
for batch in train_loader:
    show_images(batch['image'], wandb_save_name='batch_ex-train_images.png')
    show_images(batch['mask'], wandb_save_name='batch_ex-train_masks.png')
    show_images(batch['masked_image'], wandb_save_name='batch_ex-train_masked_images.png')
    break


# Torch clean up caches just in case
torch.cuda.empty_cache()

# Create the model
if config['model'] == 'Encoder-Decoder':
    model = EncoderDecoder()
elif config['model'] == 'UNet':
    model = UNet()
else:
    raise ValueError(f'Unknown model {config["model"]}')


callbacks = [
    L.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    ),
     L.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
    ),
    L.callbacks.LearningRateMonitor(logging_interval='step'),
    L.callbacks.DeviceStatsMonitor(),
    L.callbacks.ModelSummary(),
]


# Create the LightningModule
lightning_module = ImageInpainting(model, config)

# Create the trainer
trainer = L.Trainer(
    max_epochs=config['epochs'],
    callbacks=callbacks,
    logger=wandb_logger,
    # overfit_batches=1,
    )

start_training = time.time()
# Train the model
trainer.fit(lightning_module, train_loader, val_loader)
end_training = time.time()
wandb.log({'training_time': end_training - start_training})

# Calculate number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
wandb.log({'model_num_params': num_params})

# Calculate the memory it requires to store the model
num_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
wandb.log({'model_memory_size_MB': num_bytes / 1e6})


# Delete from memory train and val datasets and dataloaders to free up memory
del train_dataset, val_dataset, train_loader, val_loader


test_dir = os.path.join(DATA_DIRECTORY, 'test')
test_images = load_images_from_dir(test_dir) / 255.0
print(f'Loaded {len(test_images)} test images')


test_dataset = ShapesDataset(test_images, masks, transform=ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)


# Test the model and log the results to wandb
lightning_module.eval()
lightning_module.freeze()
trainer.test(lightning_module, test_loader)

wandb.finish()
