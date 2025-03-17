import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import timm
from transformers import BertModel, BertTokenizer
import numpy as np
import cv2
import os
import kagglehub

# Define PSAT (Plane-Slice-Aware Transformer) model
class PSAT(nn.Module):
    def __init__(self, img_channels=1, embed_dim=768):
        super(PSAT, self).__init__()
        self.backbone = timm.create_model('resnet18', pretrained=True, in_chans=img_channels, num_classes=embed_dim)
    
    def forward(self, x):
        return self.backbone(x)

# Multi-Modal Learning Model
class MultiModalModel(nn.Module):
    def __init__(self, embed_dim=768):
        super(MultiModalModel, self).__init__()
        self.img_encoder = PSAT()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(embed_dim * 2, 1)
    
    def forward(self, img, text_input_ids, text_attention_mask):
        img_features = self.img_encoder(img)
        text_features = self.text_encoder(text_input_ids, attention_mask=text_attention_mask).pooler_output
        combined = torch.cat((img_features, text_features), dim=1)
        return self.fc(combined)

# Step 1: Download Dataset using KaggleHub
path = kagglehub.dataset_download("thanhbnhphan/hc18-grand-challenge")
print("Path to dataset files:", path)

# Step 2: Locate the Correct Folder
print("Dataset Contents:", os.listdir(path))

data_path = path  # Default path
subfolders = os.listdir(path)

# If dataset is inside a subfolder (e.g., "train", "images"), update path
for folder in subfolders:
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path) and len(os.listdir(folder_path)) > 0:
        data_path = folder_path
        break

print("Final Data Path:", data_path)

# Step 3: Verify and Load Images
image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.png')]

# Stop Execution if No Images Found
if not image_paths:
    raise FileNotFoundError(f"No images found in {data_path}. Please check dataset structure.")

print(f"✅ Found {len(image_paths)} images in {data_path}")

# Dataset class for fetal ultrasound images and clinical text reports
class FetalDataset(Dataset):
    def __init__(self, image_paths, text_reports, labels):
        self.image_paths = image_paths
        self.text_reports = text_reports
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        
        text_tokens = self.tokenizer(self.text_reports[idx], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return img, text_tokens['input_ids'].squeeze(0), text_tokens['attention_mask'].squeeze(0), label

# Model training function
def train_model(model, dataloader, epochs=10, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for img, text_input_ids, text_attention_mask, label in dataloader:
            img, text_input_ids, text_attention_mask, label = img.to(device), text_input_ids.to(device), text_attention_mask.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img, text_input_ids, text_attention_mask)
            loss = loss_fn(output.squeeze(), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

if __name__ == "__main__":
    print("Training mode: Running model training...")
    # Load dataset and initialize DataLoader
    text_reports = ["Fetal head measurement normal"] * len(image_paths)
    labels = [35.0] * len(image_paths)

    dataset = FetalDataset(image_paths, text_reports, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Train the model
    model = MultiModalModel()
    train_model(model, dataloader)

    # Save the trained model
    torch.save(model.state_dict(), "C:/med3dinsight/med3dinsight_main.pth")
    print(" Model training complete and saved.")






 