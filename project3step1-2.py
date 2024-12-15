"""
AER 850 Project 3 Step 1 and 2
Created on Mon Nov 18 14:35:30 2024
@author: robstan 501095883
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO

#step 1

# Loading the image
path = "motherboard_image.jpeg"
img_real = cv2.imread(path, cv2.IMREAD_COLOR)

# Rotate the image for correct orientation
img_real_rotated = cv2.rotate(img_real, cv2.ROTATE_90_CLOCKWISE)

# Applying median blur to reduce noise while preserving edges
img_median = cv2.medianBlur(img_real_rotated, 11)

# Convert to grayscale
img_gray = cv2.cvtColor(img_median, cv2.COLOR_BGR2GRAY)

# Adaptive Thresholding for better edge separation
img_thresh = cv2.adaptiveThreshold(
    img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 10
)

# Edge detection using Canny
edges = cv2.Canny(img_thresh, 75, 250)

# Dilating edges to close gaps
edges_dilated = cv2.dilate(edges, None, iterations=12)

# Fining contours
contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Blank mask
mask = np.zeros_like(img_real_rotated)

# drawing the largest contour
largest_contour = max(contours, key=cv2.contourArea)
cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# Apply the mask to the original image
masked_img = cv2.bitwise_and(img_real_rotated, mask) 

# Displaying the mask image
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
plt.title("Mask Image")
plt.axis("off")
plt.show()

# Displaying the edge detection
plt.figure(figsize=(12, 8))
plt.imshow(edges, cmap='coolwarm')
plt.title("Edge Detection")
plt.axis("off")
plt.show()

# Displaying the final extracted image
plt.figure(figsize=(20, 14))
plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
plt.title("Final Extracted Motherboard Image")
plt.axis("off")
plt.show()

# step 2

# Initialize YOLOv8 model with pretrained weights
model = YOLO('yolov8n.pt')  

# Path to the dataset's YAML file
data_yaml_path = 'C:/Users/Robert/Desktop/data/data.yaml' 

# Train the YOLOv8 model

model.train(
    data=data_yaml_path,   
    epochs=199,           
    batch=4,           
    imgsz=1000,           
    name='pcb_component_detection',
    device=0,
    workers=0
)


