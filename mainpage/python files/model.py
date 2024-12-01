import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Machine for classification
from sklearn.metrics import accuracy_score
from joblib import dump, load  # For saving/loading models
 
# Paths
data_dir = r'C:\Users\ACER\Desktop\skin-disease-datasaet\train_set'
classes = os.listdir(data_dir)
 
# Parameters
img_height, img_width = 64, 64  # Resize images to this size
 
# Load dataset
X = []
y = []
 
for label, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_width, img_height))  # Resize to fixed size
        img = img.flatten()  # Flatten to a 1D vector
        X.append(img)
        y.append(label)
 
# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Train a model
model = SVC(kernel='linear', probability=True)  # Support Vector Classifier
model.fit(X_train, y_train)
 
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
 
# Save the model
dump(model, 'skin_disease_model.joblib')
print("Model saved as skin_disease_model.joblib")