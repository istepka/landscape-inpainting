# Define the LightningModule
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
import pytorch_lightning as L
import torch.nn as nn
import pytorch_ssim

from .utils import show_images


class ImageInpainting(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.save_hyperparameters()
        self.config = config
        
    def forward(self, mask, masked_image):
        return self.model(mask, masked_image)
    
    def training_step(self, batch, batch_idx):
        image, mask, masked_image = batch['image'], batch['mask'], batch['masked_image']
        output = self(mask, masked_image)
        loss = F.mse_loss(output, image)
        ssim_loss = 1 - pytorch_ssim.ssim(output, image)
        
        self.log(name='train_loss', value=loss, prog_bar=True)
        self.log(name='train_ssim', value=ssim_loss, prog_bar=True)
        self.log(name='train_l1', value=F.l1_loss(output, image), prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, mask, masked_image = batch['image'], batch['mask'], batch['masked_image']
        output = self(mask, masked_image)
        loss = F.mse_loss(output, image)
        l1_loss = F.l1_loss(output, image)
        
        self.log(name='val_loss', value=loss, prog_bar=True)
        self.log(name='val_l1', value=l1_loss, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        image, mask, masked_image = batch['image'], batch['mask'], batch['masked_image']
        output = self(mask, masked_image)
        
        loss = self.loss_fnc(output, image)
        
        l1_loss = F.l1_loss(output, image)
        # Similarity metric
        
        self.log(name='test_loss', value=loss)
        self.log(name='test_l1', value=l1_loss)
        
        return loss
    
    def loss_fnc(self, output, image):
        if self.config['loss'] == 'mse':
            return F.mse_loss(output, image)
        if self.config['loss'] == 'l1':
            return F.l1_loss(output, image)
        if self.config['loss'] == 'cross_entropy':
            return F.cross_entropy(output, image)
        if self.config['loss'] == 'poisson':
            return F.poisson_nll_loss(output, image)
        if self.config['loss'] == 'kldiv':
            return F.kl_div(output, image)
        
        # Default to mse
        return F.mse_loss(output, image)
    
    def configure_optimizers(self):
        if self.config['optimizer'] == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.config['learning_rate'])
        if self.config['optimizer'] == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.config['learning_rate'])
        if self.config['optimizer'] == 'adagrad':
            return torch.optim.Adagrad(self.parameters(), lr=self.config['learning_rate'])
        
        # Default to Adam
        return torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        
    
    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        
        if batch_idx == 0:
            image, mask, masked_image = batch['image'], batch['mask'], batch['masked_image']
            output = self(mask, masked_image)
            show_images(image.cpu().detach(), wandb_save_name='test_images.png', show_plot_locally=False)
            show_images(masked_image.cpu().detach(), wandb_save_name='test_masked_images.png', show_plot_locally=False)
            show_images(output.cpu().detach(), wandb_save_name='test_output.png', show_plot_locally=False)
        
        return super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)
    
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        
        if batch_idx == 0:
            image, mask, masked_image = batch['image'], batch['mask'], batch['masked_image']
            output = self(mask, masked_image)
            show_images(image.cpu().detach(), wandb_save_name=f'train_images_{self.current_epoch}.png', show_plot_locally=False)
            show_images(masked_image.cpu().detach(), wandb_save_name=f'train_masked_images_{self.current_epoch}.png', show_plot_locally=False)
            show_images(output.cpu().detach(), wandb_save_name=f'train_output_{self.current_epoch}.png', show_plot_locally=False)

        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def on_test_end(self) -> None:
        # Average the test metrics across all batches to get the final test metrics
        metrics = self.trainer.callback_metrics
        final_metrics = {}
        for k in metrics.keys():
            if 'test' in k:
                final_metrics[f'final_{k}'] = torch.stack(metrics[k]).mean()
        
        # Log the final test metrics
        self.log_dict(final_metrics)
        return super().on_test_end()


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding='same'),
            nn.BatchNorm2d(256),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding='same'),
            nn.Hardtanh(min_val=0, max_val=1),
        )
        
    def forward(self, mask, masked_image):
        x = torch.cat([mask, masked_image], dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

  
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Small unet model
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding='same'),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            # nn.BatchNorm2d(64),
        )
        
        self.encoder2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding='same'),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding='same'),
            # nn.BatchNorm2d(128),
        )
        
        self.encoder3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding='same'),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding='same'),
            # nn.BatchNorm2d(256),
        )
        
        self.bottle_neck = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding='same'),
            # nn.BatchNorm2d(256),
        )
        
        self.decoder1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, 128, 3, padding='same'),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding='same'),
            # nn.BatchNorm2d(128),
        )
        
        self.decoder2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 64, 3, padding='same'),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            # nn.BatchNorm2d(64),
        )
        
        self.decoder3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, padding='same'),
            nn.Tanh(),
        )
        
    def forward(self, mask, masked_image):
        x = torch.cat([mask, masked_image], dim=1)
        
        x1 = self.encoder1(x)
        x2 = self.encoder2(F.max_pool2d(x1, 2))
        x3 = self.encoder3(F.max_pool2d(x2, 2))
        
        x = self.bottle_neck(F.max_pool2d(x3, 2))
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder1(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder3(x)
        
        return x