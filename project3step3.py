"""
AER 850 Project 3 Step 3
Created on Mon Nov 18 14:35:30 2024
@author: robstan 501095883
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained YOLOv8 model
model = YOLO('runs/detect/pcb_component_detection6/weights/best.pt')

# Path to the test image 1
test_image_path = 'data/evaluation/ardmega.jpg'

# Run prediction
results = model.predict(source=test_image_path, imgsz=700)

# Adjust text size, font thickness, and line width to make it like your reference
image_with_results = results[0].plot(line_width=8, font_size=1.0)

# Display the output
plt.figure(figsize=(12, 10))  # Larger figure for better visuals
plt.imshow(cv2.cvtColor(image_with_results, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
plt.axis('off')  # Turn off axis for clean display
plt.show()

# Path to the test image 2
test_image_path2 = 'data/evaluation/arduno.jpg'

# Run prediction
results = model.predict(source=test_image_path2, imgsz=800)

# Adjust text size, font thickness, and line width to make it like your reference
image_with_results = results[0].plot(line_width=2, font_size=1.0)

# Display the output
plt.figure(figsize=(12, 10))  # Larger figure for better visuals
plt.imshow(cv2.cvtColor(image_with_results, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
plt.axis('off')  # Turn off axis for clean display
plt.show()

# Path to the test image 3
test_image_path3 = 'data/evaluation/rasppi.jpg'

# Run prediction
results = model.predict(source=test_image_path3, imgsz=800)

# Adjust text size, font thickness, and line width to make it like your reference
image_with_results = results[0].plot(line_width=5, font_size=1.0)

# Display the output
plt.figure(figsize=(12, 10))  # Larger figure for better visuals
plt.imshow(cv2.cvtColor(image_with_results, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
plt.axis('off')  # Turn off axis for clean display
plt.show()