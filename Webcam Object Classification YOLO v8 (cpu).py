import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def classify_image(image, texts, result = None):
    draw = ImageDraw.Draw(image)
    box_width = 150  # You can adjust the width as needed
    box_height = 100  # You can adjust the height as needed
    spacing = 20  # Spacing between boxes
    start_x = 50  # Starting x-coordinate for the boxes
    font = ImageFont.truetype("arial.ttf", 16)
    # Draw stacked boxes
    for i, text in enumerate(texts):
        box = (start_x, 50 + i * (box_height + spacing), start_x + box_width, 50 + i * (box_height + spacing) + box_height)
        draw.rectangle(box, outline="black")
        draw.text((start_x + 10, 50 + i * (box_height + spacing) + 10), text, fill="white", font = font)

    # Define text for each box
    if len(texts) == 0:
        for i in range(3):
            texts.append(result.names[result.probs.top5[i]])        
        
    return np.array(image), texts

def object_segmentation_yolo8(model, camera = 0, threshold = 0.5, font_size = 18):
    '''
    The purpose of this function is to connect to a webcam or external camera
    such as a GoPro and do object segmentation using the YOLO v8 model. 
    
    Parameters
    ----------
    camera : Int, optional
        The type of camera. 0 is the built in webcam, 1-n is for other webcams.
        The default is 0.
    
    '''
    cap = cv2.VideoCapture(camera)
    
    if not cap.isOpened():
        print("Cannot open camera. Aborting.")
        exit()
    else:
        print("Camera opened. Attempting to retrieve frame.")
    
    frame_counter = 0
    texts = []
    while True:
        ret, frame = cap.read()
    
        if not ret:
            print("Error retrieving frame. Aborting.")
            break
    
        
        image_pil = Image.fromarray(frame)
        if frame_counter % 200 == 0:
            texts = []
            result = model(frame)[0]
            output, texts = classify_image(image_pil, texts, result = result)
            cv2.imshow('object detection using YOLO', output)
        else:
            output, texts = classify_image(image_pil, texts = texts)
            cv2.imshow('object detection using YOLO', output)
            
        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#model sizes:
#   - Nano: yolov8n
#   - Small: yolov8s
#   - Medium: yolov8m
#   - Large: yolov8l
#   - XLarge: yolov8x

model = YOLO('yolov8m-cls.pt')
object_segmentation_yolo8(model = model)    