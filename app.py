from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

# Load your PyTorch model
# Replace 'YourModelClass' and 'your_model.pth' with your actual model class and path
from src.unet_models import UNet

model = UNet()
model.load_state_dict(torch.load('your_model.pth'))
model.eval()

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        img = Image.open(file).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(img)

        # Process the output as needed
        # ...

        return render_template('index.html', message='Prediction complete')

if __name__ == '__main__':
    app.run(debug=True)
