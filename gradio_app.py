import gradio as gr
import numpy as np
from PIL import Image
import os 
from src.utils import create_circle, create_square, paste_shape
from src.models import ImageInpainting, UNet
from src.datasets import ShapesDataset, ToTensor
import torch

def load_images(dir: str = 'images') -> np.ndarray:
    images = []
    for filename in os.listdir(dir):
        if filename.endswith('.jpg'):
            images.append(os.path.join(dir, filename))
    images = [Image.open(image).resize((256, 256)) for image in images]
    return np.array(images)

def process_image(image, mask):
    
    # Make sure the image is in RGB format
    image = Image.fromarray(image).convert('RGB')
    # Make sure it's the right size
    image = image.resize((256, 256))
    mask = Image.fromarray(mask).convert('RGB')
    mask = mask.resize((256, 256))
    
    
    # Paste the mask onto the image
    mask = np.array(mask) / 255.0
    image = np.array(image) / 255.0
    
    masked_image = image * (1 - mask)
    print(mask.min(), mask.max())
    print(image.min(), image.max())
    print(masked_image.min(), masked_image.max())
    
    masked_image = torch.from_numpy(masked_image).float().permute(2, 0, 1)
    mask = torch.from_numpy(mask).float().permute(2, 0, 1)
    
    inpainted = model(mask.unsqueeze(0), masked_image.unsqueeze(0))
    
    # Convert the output to numpy array
    output = inpainted.squeeze(0).permute(1, 2, 0).detach().numpy()
    masked_image = masked_image.permute(1, 2, 0).detach().numpy()
    
    return (masked_image * 255).astype(np.uint8), (output  * 255).astype(np.uint8)

model = ImageInpainting.load_from_checkpoint(checkpoint_path='checkpoints/inference.ckpt', model=UNet(), config=None)
model.eval()

example_images = load_images()
example_masks_elements = [create_circle(32), create_square(32)]

example_masks = np.zeros_like(example_images)
for i in range(len(example_images)):
    example_masks[i] = paste_shape(np.zeros_like(example_images[0]), example_masks_elements[i % 2]) * 255


demo = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(), gr.Image()],
    outputs=[gr.Image(), gr.Image()],
    examples=[
        [example_images[0], example_masks[0]],
        [example_images[1], example_masks[1]],
    ],
    
)

demo.launch()
