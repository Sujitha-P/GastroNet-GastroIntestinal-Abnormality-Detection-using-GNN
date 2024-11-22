import os
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from skimage import io
from skimage.transform import resize
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix   
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define the base directory
base_dir = r'E:\DL'
kvasir_folder = os.path.join(base_dir, r"E:\DL\kvasir-dataset-v2")
kvasir_features_folder = os.path.join(base_dir, r"E:\DL\kvasir-dataset-v2-features")

# Function to load images
def load_kvasir_images(kvasir_folder, target_size=(224, 224)):
    images, labels = [], []
    for disease_folder in os.listdir(kvasir_folder):
        disease_path = os.path.join(kvasir_folder, disease_folder)
        if os.path.isdir(disease_path):
            for img_file in os.listdir(disease_path):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(disease_path, img_file)
                    image = io.imread(img_path, as_gray=True)
                    image_resized = resize(image, target_size, anti_aliasing=True)
                    images.append(image_resized)
                    labels.append(disease_folder)
    return images, labels

# Function to load features
def load_kvasir_features(kvasir_features_folder):
    features = []
    for disease_folder in os.listdir(kvasir_features_folder):
        disease_path = os.path.join(kvasir_features_folder, disease_folder)
        if os.path.isdir(disease_path):
            for feature_file in os.listdir(disease_path):
                if feature_file.endswith('.features'):
                    feature_path = os.path.join(disease_path, feature_file)
                    with open(feature_path, 'r') as f:
                        feature_data = [float(value) for value in f.readline().split(',')[1:]]
                    features.append(feature_data)
    return features


kvasir_images, kvasir_labels = load_kvasir_images(kvasir_folder)
kvasir_features = load_kvasir_features(kvasir_features_folder)

# Create Graph Data Function
def create_graph_data(images, features, labels):
    data_list = []
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    for i in range(len(images)):
        x = torch.tensor(features[i], dtype=torch.float).view(1, -1)
        edge_index = torch.tensor([[i, j] for j in range(len(images)) if j != i], dtype=torch.long).t().contiguous()
        label_index = labels_encoded[i]
        data = Data(x=x, edge_index=edge_index, y=torch.tensor(label_index, dtype=torch.long).view(1, -1))
        data_list.append(data)
    return data_list, label_encoder

# Define the GNN model
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

# Initialize Model
num_features = len(kvasir_features[0])
num_classes = len(set(kvasir_labels))
model = GNNModel(num_features, num_classes)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_model(graph_data, model, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in graph_data:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(graph_data):.4f}")

# Evaluate model
def evaluate_model(graph_data, model):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for data in graph_data:
            out = model(data)
            pred = out.argmax(dim=1)
            all_predictions.append(pred.item())
            all_labels.append(data.y.item())
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    


# Run training and evaluation on the combined dataset
def run_combined_train_test(kvasir_images, kvasir_labels, kvasir_features, model, criterion, optimizer, num_epochs=50):
    graph_data, label_encoder = create_graph_data(kvasir_images, kvasir_features, kvasir_labels)
    train_data, test_data = train_test_split(graph_data, test_size=0.2, stratify=kvasir_labels, random_state=42)
    
    print("\nTraining on combined data:")
    train_model(train_data, model, criterion, optimizer, num_epochs)
    
    print("\nEvaluating on test data:")
    evaluate_model(test_data, model)

run_combined_train_test(kvasir_images, kvasir_labels, kvasir_features, model, criterion, optimizer, num_epochs=50)

# Save the model once
save_model_path = os.path.join(base_dir, "gnn_combined_model.pkl")
torch.save(model.state_dict(), save_model_path)
print(f"Model saved to {save_model_path}")

