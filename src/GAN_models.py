import lightning as L
from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F


class GAN(LightningModule):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(input_dim, hidden_dim, output_dim)
        self.discriminator = Discriminator(input_dim, hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def training_step(self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        x, y = batch
        z = torch.randn(x.shape[0], self.hparams.input_dim)
        if optimizer_idx == 0:
            # Train Generator
            fake_y = self(z)
            g_loss = self.generator.loss(fake_y, y)
            self.log('g_loss', g_loss)
            return g_loss
        elif optimizer_idx == 1:
            # Train Discriminator
            fake_y = self(z).detach()
            d_loss = self.discriminator.loss(fake_y, y)
            self.log('d_loss', d_loss)
            return d_loss

    def configure_optimizers(self) -> tuple:
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []
    
    
class Generator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.MSELoss()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

    def loss(self, fake_y: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss(fake_y, y)