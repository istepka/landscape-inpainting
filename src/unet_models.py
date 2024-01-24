import lightning as L
from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow

class EncoderDecoder(LightningModule):
    def __init__(self, hparams):
        super(EncoderDecoder, self).__init__()

        self.params = hparams

        # Define the encoder layers
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding='same'),
            nn.Sigmoid(),
        )
        
    def forward(self, x, mask):
        return self.model(x * mask)
        

    def training_step(self, batch, batch_idx):
        # Unpack the batch
        image, mask = batch
        
        # Generate the output
        output = self(image, mask)
        
        # Calculate the loss
        loss = F.binary_cross_entropy(output, image)
        
        # Calculate L1
        l1 = F.l1_loss(output, image)
        rmse = torch.sqrt(F.mse_loss(output, image))
        
        # Log the loss
        self.log('train_loss', loss)
        self.log('train_l1', l1)
        self.log('train_rmse', rmse)
        
        mlflow.log_metric('train_loss', loss)
        mlflow.log_metric('train_l1', l1)
        mlflow.log_metric('train_rmse', rmse)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Unpack the batch
        image, mask = batch
        
        # Generate the output
        output = self(image, mask)
        
        # Calculate the loss
        loss = F.binary_cross_entropy(output, image)
        
        # Log the loss
        self.log('val_loss', loss)
        self.log('val_l1', F.l1_loss(output, image))
        self.log('val_rmse', torch.sqrt(F.mse_loss(output, image)))
        
        mlflow.log_metric('val_loss', loss)
        mlflow.log_metric('val_l1', F.l1_loss(output, image))
        mlflow.log_metric('val_rmse', torch.sqrt(F.mse_loss(output, image)))
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # Unpack the batch
        image, mask = batch
        
        # Generate the output
        output = self(image, mask)
        
        # Calculate the loss
        loss = F.binary_cross_entropy(output, image)
        
        # Log the loss
        self.log('test_loss', loss)
        self.log('test_l1', F.l1_loss(output, image))
        self.log('test_rmse', torch.sqrt(F.mse_loss(output, image)))
        
        mlflow.log_metric('test_loss', loss)
        mlflow.log_metric('test_l1', F.l1_loss(output, image))
        mlflow.log_metric('test_rmse', torch.sqrt(F.mse_loss(output, image)))
        
        return loss
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.params['lr'], 
            weight_decay=self.params['weight_decay']
        )
        return optimizer
    
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        mlflow.log_metric('lr', self.optimizers().param_groups[0]['lr'])


class UNet(LightningModule):
    def __init__(self, hparams):
        super(UNet, self).__init__()

        self.params = hparams

        # Define the encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        # Define the decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding='same'),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        # Encode the input
        x1 = self.encoder[0:3](x * mask)
        x2 = self.encoder[4:7](x1)
        x3 = self.encoder[8:](x2)

        # Decode the encoded features with skip connections
        decoded = self.decoder[0:2](x3)
        decoded = torch.cat([x2, decoded], dim=1)
        decoded = self.decoder[2:4](decoded)
        decoded = torch.cat([x1, decoded], dim=1)
        decoded = self.decoder[4:](decoded)

        return decoded
        

    def training_step(self, batch, batch_idx):
        # Unpack the batch
        image, mask = batch
        
        # Generate the output
        output = self(image, mask)
        
        # Calculate the loss
        loss = F.binary_cross_entropy(output, image)
        
        # Calculate L1
        l1 = F.l1_loss(output, image)
        rmse = torch.sqrt(F.mse_loss(output, image))
        
        # Log the loss
        self.log('train_loss', loss)
        self.log('train_l1', l1)
        self.log('train_rmse', rmse)
        
        mlflow.log_metric('train_loss', loss)
        mlflow.log_metric('train_l1', l1)
        mlflow.log_metric('train_rmse', rmse)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Unpack the batch
        image, mask = batch
        
        # Generate the output
        output = self(image, mask)
        
        # Calculate the loss
        loss = F.binary_cross_entropy(output, image)
        
        # Log the loss
        self.log('val_loss', loss)
        self.log('val_l1', F.l1_loss(output, image))
        self.log('val_rmse', torch.sqrt(F.mse_loss(output, image)))
        
        mlflow.log_metric('val_loss', loss)
        mlflow.log_metric('val_l1', F.l1_loss(output, image))
        mlflow.log_metric('val_rmse', torch.sqrt(F.mse_loss(output, image)))
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # Unpack the batch
        image, mask = batch
        
        # Generate the output
        output = self(image, mask)
        
        # Calculate the loss
        loss = F.binary_cross_entropy(output, image)
        
        # Log the loss
        self.log('test_loss', loss)
        self.log('test_l1', F.l1_loss(output, image))
        self.log('test_rmse', torch.sqrt(F.mse_loss(output, image)))
        
        mlflow.log_metric('test_loss', loss)
        mlflow.log_metric('test_l1', F.l1_loss(output, image))
        mlflow.log_metric('test_rmse', torch.sqrt(F.mse_loss(output, image)))
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.params['lr'], 
            weight_decay=self.params['weight_decay']
        )
        return optimizer
    
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        mlflow.log_metric('lr', self.optimizers().param_groups[0]['lr'])
