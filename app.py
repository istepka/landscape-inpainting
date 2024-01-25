from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import torch
import torchvision.transforms as transforms
from src.models import ImageInpainting, UNet
from src.utils import create_square
from src.datasets import ShapesDataset, ToTensor
import base64

app = Flask(__name__)

# Load the pre-trained PyTorch model
model = ImageInpainting.load_from_checkpoint(checkpoint_path='checkpoints/inference.ckpt', model=UNet(), config=None)
model.eval()

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((256, 256))
])

# API endpoint to process the uploaded image and perform inpainting
@app.route('/inpaint', methods=['POST'])
def inpaint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Read the image from the request
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    image_np = np.array(image) / 255.0

    # Prepare the image for inpainting
    dataset = ShapesDataset(images=np.array([image_np]), masks=np.array([create_square(32)]), augument=False)
    image = dataset[0]['image']
    image = ToTensor()(dataset[0])['image']

    # Perform inpainting
    inpainted_image = inpaint_image(image)

    # Convert the inpainted image to PIL image
    inpainted_image_pil = Image.fromarray(np.uint8(inpainted_image * 255))

    # Convert the PIL image to bytes and then to base64
    buffered = io.BytesIO()
    inpainted_image_pil.save(buffered, format="JPEG")
    inpainted_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({'inpainted_image': inpainted_image_base64}), 200

# Function to perform inpainting
def inpaint_image(image):
    # Perform inpainting
    mask = create_square(32)
    masked_image = image * (1 - mask)
    output = model(mask.unsqueeze(0), masked_image.unsqueeze(0))
    output = output.squeeze(0)

    # Convert the output to numpy array
    output = output.detach().numpy()

    return output

if __name__ == '__main__':
    app.run(debug=True)
