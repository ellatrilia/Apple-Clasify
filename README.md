# Apple-Clasify
This tutorial explains how to train, detect, and classify apples based on color (red, yellow, green) using a pre-trained YOLOv8 model.
<img src="https://drive.google.com/uc?id=1cFpPtUpSlOiXk5XhoEwEZvI2vIUIGtdU" width="100%" height="300 cm">

## Prepare Dataset
   * Source dataset: [Apple Annotation with Roboflow](https://universe.roboflow.com/tugas-akhir-70fw5/apel-mrg3l/dataset/1).
   * The dataset has been annotated using the Roboflow platform.
   * Augmentations applied to improve model robustness:
     * Flip: Horizontal, Vertical
     * 90° Rotate: Clockwise, Counter-Clockwise
     * Rotation: Between -15° and +15
     * Shear: ±10° Horizontal, ±10° Vertical
     * Saturation: Between -25% and +25%
     * Exposure: Between -10% and +10%
     * Blur: Up to 2.5px
     * Noise: Up to 0.1% of pixels
       
   * Download dataset using Roboflow API:
     ```
     !pip install roboflow
     from roboflow import Roboflow
     rf = Roboflow(api_key="38tP3MAn9Msvn367ZMee")
     project = rf.workspace("ella-trilia-oviana").project("apel-mrg3l-dfvxm-kdx3d")
     version = project.version(1)
     dataset = version.download("yolov8")
      ```

## Train the Dataset Using YOLOv8 Pre-Trained Model

Link to train: [Notebook Train](https://colab.research.google.com/github/ellatrilia/Train-Custom-Dataset-With-YOLOv8-Pre-Trained-Model/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb#scrollTo=D2YkphuiaE7_)
  * Run the training command with YOLOv8:
    ```
    !yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=100 imgsz=800 patience=0 plots=True
    ```
    This command will:
    * Train a YOLOv8 object detection model
    * Use yolov8s.pt as the pre-trained model
    * Train for 100 epochs
    * Use an image size of 800px
    * Set patience to 0 (no early stopping)
    * Generate training plots

  * Download Trained Model
    After training is complete, the best model weights will be saved as best.pt in the Colab Files section. You can download it using:
    ```
    from google.colab import files
    files.download('runs/detect/train/weights/best.pt')
    ```
    **Now, you have a trained YOLOv8 model ready to be used for inference! 🚀**
    
# Detection & Classification (Jupyter Notebook)
This repository provides a Python script to classify apples based on their color (red, yellow, green) using a fine-tuned YOLOv8 model. The script detects apples in an image, crops them, and saves them with the corresponding color label.

## Requirements
Make sure you have the following dependencies installed:
```
pip install ultralytics opencv-python numpy ipython
```
## Usage
1. Modify the Paths:
   * Update the model path in the script to the location of your trained YOLOv8 model (best.pt).
   * Update the image path to the location of your test image.
2. Run the Script in Jupyter Notebook
Save the following script as a Jupyter Notebook (.ipynb) and execute it cell by cell.

##  Script Overview
Save the following code as a cell in your Jupyter Notebook:
```
import cv2
import numpy as np
import os
from ultralytics import YOLO
from IPython.display import display, Image

# Load YOLOv8 model with custom dataset
model = YOLO(r"C:\Users\asus\Downloads\best.pt")  # Update path accordingly

# Load image
image_path = r"C:\Users\asus\Downloads\classify.jpg"  # Update path accordingly
image = cv2.imread(image_path)

# Perform detection
results = model(image)

def classify_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "red": [(np.array([0, 120, 70]), np.array([10, 255, 255])),
                 (np.array([170, 120, 70]), np.array([180, 255, 255]))],
        "yellow": [(np.array([20, 150, 150]), np.array([35, 255, 255]))],
        "green": [(np.array([36, 100, 100]), np.array([85, 255, 255]))]
    }
    
    scores = {color: sum(np.sum(cv2.inRange(hsv, lower, upper)) for lower, upper in ranges)
              for color, ranges in color_ranges.items()}
    
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "unknown"

# Create output directories
output_dirs = {color: os.makedirs(color + "_apples", exist_ok=True) for color in ["red", "yellow", "green"]}
count = {"red": 0, "yellow": 0, "green": 0}
detected_images = []

# Process detections
for box in results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)
    cropped_apple = image[y1:y2, x1:x2]
    color = classify_color(cropped_apple)
    
    if color in count:
        count[color] += 1
        output_path = os.path.join(f"{color}_apples", f"{color}_{count[color]}.jpg")
        cv2.imwrite(output_path, cropped_apple)
        detected_images.append(output_path)

# Display detected images
for img_path in detected_images:
    display(Image(img_path, width=150))
    print(os.path.basename(img_path))
```
## Output
* The detected apples will be saved in separate folders (red_apples/, yellow_apples/, green_apples/).
* The filenames will follow the format {color}_{number}.jpg (e.g., red_1.jpg).
  
<img class="size-full wp-image-8990" 
     src="https://drive.google.com/uc?export=view&id=1Q4IC4y4rKfLTeAIsMjOUbe13l0zpFXbd" 
     alt="Gambar 1" 
     width="200" height="200">
<img class="size-full wp-image-8991" 
     src="https://drive.google.com/uc?export=view&id=1rFODLqYWlT3ZGow71CqwZixRf4yQKARn" 
     alt="Gambar 2" 
     width="200" height="200">
<img class="size-full wp-image-8992" 
     src="https://drive.google.com/uc?export=view&id=144Z9-D4aZYZUfoIF0ZMiRrErH9Y9HOdL" 
     alt="Gambar 3" 
     width="200" height="200">

## Notes
* Ensure that the paths to the model and image are correctly set.
* The color classification is based on HSV color range thresholds.

## License
This project is open-source and can be freely modified and distributed.






   
   
     
   

