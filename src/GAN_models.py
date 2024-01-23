import lightning as L
from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple_GAN(LightningModule):
    '''
    This is a simple GAN model for image inpainting. 
    How it works:
        1. The generator takes in the original image and the mask, and outputs the inpainted image.
        2. The discriminator takes in the original image and the inpainted image, and outputs a score.
        3. The generator and discriminator are trained in an adversarial manner.
        4. The generator is also trained to minimize the L1 loss between the inpainted image and the original image.
        
    '''
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.generator = CNN_Bottleneck_Generator(hparams)
        self.discriminator = CNN_Discriminator(hparams)
        
        # Losses
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, x, mask):
        # Generate inpainted image
        inpainted_image = self.generator(x, mask)
        
        # Score the inpainted image
        score = self.discriminator(inpainted_image)
        
        return inpainted_image, score
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # Unpack batch
        x, mask = batch
        
        # Train generator
        if optimizer_idx == 0:
            # Generate inpainted image
            inpainted_image = self.generator(x, mask)
            
            # Score the inpainted image
            score = self.discriminator(inpainted_image)
            
            # Calculate loss
            g_loss = self.bce_loss(score, torch.ones_like(score)) + self.l1_loss(inpainted_image, x)
            
            # Log loss
            self.log('g_loss', g_loss)
            
            return g_loss
        
        # Train discriminator
        if optimizer_idx == 1:
            # Generate inpainted image
            inpainted_image = self.generator(x, mask)
            
            # Score the inpainted image
            score = self.discriminator(inpainted_image)
            
            # Calculate loss
            d_loss = self.bce_loss(score, torch.ones_like(score)) + self.bce_loss(self.discriminator(x), torch.zeros_like(score))
            
            # Log loss
            self.log('d_loss', d_loss)
            
            return d_loss
        
    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.generator_lr, weight_decay=self.hparams.weight_decay)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.discriminator_lr, weight_decay=self.hparams.weight_decay)
        return optimizer_G, optimizer_D


class CNN_Bottleneck_Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        filters = hparams.generator_filters
        
        assert filters[0] == 3, 'First filter size must be 3 (RGB)'
        assert filters[-1] == 3, 'Last filter size must be 3 (RGB)'
        
        self.layers = []
        # Encoder - Decoder
        for i in range(len(filters)-1, 1):
            self.layers.append(nn.Conv2d(filters[i-1], filters[i], kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.BatchNorm2d(filters[i]))
            self.layers.append(nn.LeakyReLU(0.2))
            
        # Add tanh which is applied at the end
        self.layers.append(nn.Tanh())
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x, mask):
        # Concatenate the mask to the input
        x = torch.cat([x, mask], dim=1)
        return self.model(x)
    
class CNN_Discriminator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        filters = hparams.discriminator_filters
        
        assert filters[0] == 3, 'First filter size must be 3 (RGB)'
        assert filters[-1] == 1, 'Last filter size must be 1 (score)'
        
        self.layers = []
        # Just a CNN
        for i in range(len(filters)-1, 1):
            self.layers.append(nn.Conv2d(filters[i-1], filters[i], kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.BatchNorm2d(filters[i]))
            
            # Add maxpool and leaky relu after every layer except the last one
            if i != len(filters)-1:
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
                self.layers.append(nn.LeakyReLU(0.2))
            
        # Add sigmoid which is applied at the end
        self.layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.model(x)

        
        

   