import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
import torch
from torch_geometric.data import Data
from PIL import Image
from skimage.transform import resize
import torch.nn as nn
import torch.nn.functional as F

# Initialize Flask app
app = Flask(__name__)

class GNNModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = nn.Linear(num_features, 64)
        self.conv2 = nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Model and paths
base_dir = r'E:\DL'
model_path = os.path.join(base_dir, r"E:\DL\gnn_combined_model.pkl")


# Set model parameters
num_features = 167 
num_classes = 8   
model = GNNModel(num_features, num_classes)  
model.load_state_dict(torch.load(model_path))  
model.eval()


# Preprocess image function
def preprocess_image(image, target_size=(224, 224)):
    image_resized = resize(np.array(image), target_size, anti_aliasing=True)
    image_flattened = image_resized.flatten()
    image_flattened = image_flattened[:167]  #
    return torch.tensor(image_flattened, dtype=torch.float32).unsqueeze(0)


class_names = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis"
]


# Prediction function
def predict(image_path):
    image = Image.open(image_path).convert('L')
    processed_image = preprocess_image(image)
    x = processed_image.view(1, -1)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    with torch.no_grad():
        output = model(data)
        predicted_class_index = output.argmax(dim=1).item()

    # Map the predicted index to the corresponding class name
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Flask routes
@app.route('/')
def login():
    return render_template('login.html')  
@app.route('/index')
def index():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'})

    file_path = os.path.join(base_dir, 'temp_image.jpg')
    file.save(file_path)

    try:
        prediction = predict(file_path)
        return jsonify({'prediction': str(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        os.remove(file_path)


if __name__ == '__main__':
    app.run(debug=True)
