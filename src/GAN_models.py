import lightning as L
from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


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
        self.params = hparams
        self.generator = CNN_Bottleneck_Generator(hparams)
        self.discriminator = CNN_Discriminator(hparams)
        
        # Losses
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        
        self.automatic_optimization = False
        
    def forward(self, x, mask):
        # Generate inpainted image
        x_cat_mask = torch.cat([x, mask], dim=1)
        inpainted_image = self.generator(x_cat_mask)
        
        # Score the inpainted image
        score = self.discriminator(inpainted_image)
        
        return inpainted_image, score
    
    def training_step(self, batch, batch_idx):
        # Unpack batch
        x, mask = batch
        
        g_opt, d_opt = self.optimizers()
        
        # ---------------
        # Train generator
        # ---------------
        # Generate inpainted image
        x_cat_mask = torch.cat([x, mask], dim=1)
        g_x = self.generator(x_cat_mask)
        # Score the inpainted image
        score = self.discriminator(g_x)

        g_pred_loss = self.bce_loss(score, torch.ones_like(score))
        g_fidelity_loss = self.l1_loss(g_x, x)
        g_loss = g_pred_loss + g_fidelity_loss
        
        g_opt.zero_grad()
        self.manual_backward(g_loss, retain_graph=True)
        g_opt.step()
        
        # ---------------
        # Train discriminator
        # ---------------
        fake_loss = self.bce_loss(score, torch.zeros_like(score))
        real_loss = self.bce_loss(self.discriminator(x), torch.ones_like(score))
        
        d_loss = fake_loss + real_loss
        
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()
        
        self.log_dict({
            'g_loss': g_loss,
            'd_loss': d_loss
        }, on_step=True, on_epoch=True, prog_bar=True)

        
    def validation_step(self, batch, batch_idx):
        # Unpack batch
        x, mask = batch
        
        # Generate inpainted image
        x_cat_mask = torch.cat([x, mask], dim=1)
        g_x = self.generator(x_cat_mask)
        # Score the inpainted image
        score = self.discriminator(g_x)
        
        g_pred_loss = self.bce_loss(score, torch.ones_like(score))
        g_fidelity_loss = self.l1_loss(g_x, x)
        g_loss =  g_pred_loss +  g_fidelity_loss
        
        fake_loss = self.bce_loss(score, torch.zeros_like(score))
        real_loss = self.bce_loss(self.discriminator(x), torch.ones_like(score))
        
        d_loss = fake_loss + real_loss
        
        self.log_dict({
            'val_g_loss': g_loss,
            'val_d_loss': d_loss
        }, on_step=True, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(
            self.generator.parameters(), 
            lr=self.params['generator_lr'], 
            weight_decay=self.params['weight_decay']
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.params['discriminator_lr'], 
            weight_decay=self.params['weight_decay']
        )
        return optimizer_G, optimizer_D



class CNN_Bottleneck_Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.params = hparams
        filters = hparams['generator_filters']
        
        assert filters[0] == 3, 'First filter size must be 3 (RGB)'
        
        self.layers = []
        # CNN encoder + linear layer
        for i in range(len(filters)-1, 1):
            self.layers.append(nn.Conv2d(filters[i-1], filters[i], kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.BatchNorm2d(filters[i]))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
            self.layers.append(nn.LeakyReLU(0.2))
            
        self.layers.append(nn.Flatten())
        self.layers.append(nn.LazyLinear(256))
        self.layers.append(nn.Linear(256, filters[0] * self.params['pixels']**2))
        # Add tanh which is applied at the end
        self.layers.append(nn.Tanh())
        # Reshape to image
        self.layers.append(nn.Unflatten(1, (filters[0], self.params['pixels'], self.params['pixels'])))
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        # Concatenate the mask to the input
        
        return self.model(x)
    
class CNN_Discriminator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.params = hparams
        filters = hparams['discriminator_filters']
        
        assert filters[0] == 3, 'First filter size must be 3 (RGB)'
        
        self.layers = []
        # Just a CNN
        for i in range(len(filters)-1, 1):
            self.layers.append(nn.Conv2d(filters[i-1], filters[i], kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.BatchNorm2d(filters[i]))
            
            # Add maxpool and leaky relu after every layer except the last one
            if i != len(filters)-1:
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
                self.layers.append(nn.LeakyReLU(0.2))
            
        self.layers.append(nn.Flatten())
        self.layers.append(nn.LazyLinear(1))
        # Add sigmoid which is applied at the end
        self.layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.model(x)

        
        

   