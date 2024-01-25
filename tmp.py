from torchviz import make_dot
from src.models import EncoderDecoder, UNet
import torch

# Initialize the model
encoder_decoder = EncoderDecoder()

# Generate a random input
mask = torch.randn(1, 3, 32, 32)  # Assuming input size is 32x32
masked_image = torch.randn(1, 3, 32, 32)  # Assuming input size is 32x32

# Pass the input through the model
output = encoder_decoder(mask, masked_image)

# Generate a visualization of the model
dot = make_dot(output, params=dict(encoder_decoder.named_parameters()), show_attrs=True, show_saved=True)
dot.render("EncoderDecoder", format="png")  # Save the visualization as a PNG file
