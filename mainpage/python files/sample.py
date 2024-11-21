import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Paths for training and test sets
train_set_path = r"C:\Users\ACER\Desktop\skin-disease-datasaet\train_set"
test_set_path = r"C:\Users\ACER\Desktop\skin-disease-datasaet\test_set"

# Function to preprocess a single image
import cv2
def preprocess_image(image_path, target_size=(250, 250)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0
    return img_normalized

# Function to preprocess a dataset
def preprocess_dataset(data_path, target_size=(250, 250)):
    images, labels = [], []
    class_names = os.listdir(data_path)
    for class_index, class_name in enumerate(class_names):
        class_folder = os.path.join(data_path, class_name)
        for image_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_name)
            img = preprocess_image(image_path, target_size)
            if img is not None:
                images.append(img)
                labels.append(class_index)
    return np.array(images), np.array(labels), class_names

# Preprocess datasets
X_train, y_train, train_classes = preprocess_dataset(train_set_path)
X_test, y_test, test_classes = preprocess_dataset(test_set_path)

# Convert datasets to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network
class SkinDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SkinDiseaseClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 62 * 62, 128)  # Adjust for input size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = SkinDiseaseClassifier(num_classes=len(train_classes))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):  # Adjust epochs as needed
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Evaluate the model
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

# Save the model
torch.save(model.state_dict(), "skin_disease_model.pth")

# To load the model later:
# model.load_state_dict(torch.load("skin_disease_model.pth"))
