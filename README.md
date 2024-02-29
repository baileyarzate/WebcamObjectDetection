# Object Detection using YOLOv8, YOLOv5, and Faster R-CNN
This repository contains scripts for performing object detection using the YOLOv5, YOLOv8, and Faster R-CNN models. The scripts are capable of detecting objects in real-time from live video streams captured by a webcam or an external camera.

## YOLOv8 Object Detection

The `Webcam Object Detection YOLOv8 (cpu).py` script utilizes the YOLOv8 model for object detection. It connects to a webcam or an external camera and performs real-time object recognition. The script draws bounding boxes around detected objects along with labels indicating the type of object and its confidence score.

### Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- PIL (`Pillow`)
- Ultralytics YOLO (`ultralytics`)

### Installation

1. Install required Python packages:
```bash
pip install opencv-python numpy Pillow ultralytics
```

2. Install the YOLO v8 model:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
```

3. Usage
```bash
python Webcam Object Detection YOLOv8 (cpu).py
```
By default, the script will use the built-in webcam (camera index 0) for object detection. Press 'q' to exit the program.

Parameters
- camera (int, optional): The camera index. Use 0 for the built-in webcam, or specify 1-n for other connected cameras. Default is 0.
- threshold (float, optional): Confidence threshold for object detection (between 0 and 1). Default is 0.5.
- font_size (int, optional): Font size for object labels. Default is 18.

## YOLO v5 Object Detection

The `Webcam Object Detection YOLOv5 (cpu).py` script utilizes the YOLOv5 model for object detection. It connects to a webcam or an external camera and performs real-time object recognition. The script draws bounding boxes around detected objects along with labels indicating the type of object and its confidence score.

### Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Torch (`torch`)
- Ultralytics YOLO (`ultralytics`)

### Installation

1. Install required Python packages:
```bash
pip install opencv-python numpy torch torchvision
```

2. Install the YOLOv5 model:
```python
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
```
3. Usage
```bash
python Webcam Object Detection YOLOv5 (cpu).py
```
By default, the script will use the built-in webcam (camera index 0) for object detection. Press 'q' to exit the program.

Parameters
- camera (int, optional): The camera index. Use 0 for the built-in webcam, or specify 1-n for other connected cameras. Default is 0.
- threshold (float, optional): Confidence threshold for object detection (between 0 and 1). Default is 0.5.
- font_size (int, optional): Font size for object labels. Default is 18.

## Faster R-CNN Object Detection

The `PyTorch Webcam Object Detection.py` script utilizes the Faster R-CNN model for object detection. It connects to a webcam or an external camera and performs real-time object recognition. The script draws bounding boxes around detected objects along with labels indicating the type of object and its confidence score.
### Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy (numpy)
- TorchVision (torchvision)
- PyCOCO Tools (pycocotools)

1. Install required Python packages:
```bash
pip install opencv-python numpy torchvision pycocotools
```
By default, the script will use the built-in webcam (camera index 0) for object detection. Press 'q' to exit the program.

## Acknowledgments
This project utilizes the Ultralytics YOLO library for YOLOv8 object detection and TorchVision for Faster R-CNN object detection.
YOLO (You Only Look Once) and Faster R-CNN are state-of-the-art, real-time object detection systems that identify multiple objects in images or video streams.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
