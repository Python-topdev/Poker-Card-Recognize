import cv2
import numpy as np
import os

# Set the path to your dataset
dataset_path = 'New_Data'
categories = os.listdir(dataset_path)

# Initialize lists to hold features and labels
features = []
labels = []

for label, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)
    
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        
        img = np.array(cv2.imread(image_path))
        
        img=cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        img_resized = cv2.resize(img, (50, 50))
        _,img_resized=cv2.threshold(img_resized, 200, 255, cv2.THRESH_BINARY)
        img_flattened = img_resized.flatten()
        
        features.append(img_flattened)
        labels.append(label)

# Convert lists to NumPy arrays
X = np.array(features, dtype=np.float32)  # Ensure features are float32
y = np.array(labels, dtype=np.int32)      # Ensure labels are int32

# Create and train the k-NN model
knn = cv2.ml.KNearest_create()
knn.train(X, cv2.ml.ROW_SAMPLE, y)

# Optionally, save the model
knn.save('knn_model.xml')
