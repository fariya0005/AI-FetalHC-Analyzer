import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from med3dinsight_main import MultiModalModel  # Import model class from main script

# Load the trained model
model_path = "C:/med3dinsight/med3dinsight_main.pth"
model = MultiModalModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_HC(image_path, text_report):
    """Predict Head Circumference from ultrasound image and clinical text"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    text_tokens = tokenizer(text_report, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

    with torch.no_grad():
        hc_prediction = model(img, text_tokens['input_ids'], text_tokens['attention_mask'])

    return hc_prediction.item()

# Example: Predict on a test image
test_image = "C:/med3dinsight/test_set/004_HC.png"  # Updated path
test_text = "Fetal head measurement normal"

predicted_hc = predict_HC(test_image, test_text)
print(f"\U0001F4D6 Predicted Head Circumference: {predicted_hc} mm")

# Display the image
img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title(f"Predicted HC: {predicted_hc} mm")
plt.axis("off")
plt.show()

# Compare with actual HC from CSV
csv_path = "C:/med3dinsight/test_set_pixel_size.csv"  # Updated path
df = pd.read_csv(csv_path)

# Normalize filename formatting
df["filename"] = df["filename"].str.lower().str.strip()
image_name = "004_HC.png".lower().strip()

# Print available filenames to check if the file exists
print("CSV Filenames:", df["filename"].tolist())

# Check if the image exists in the CSV and get actual HC value
if image_name in df["filename"].values:
    actual_hc = df[df["filename"] == image_name]["HC"].values[0]
else:
    actual_hc = "Not Available"

print(f" Actual HC from CSV: {actual_hc} mm")



