import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def draw_bounding_boxes(image, boxes, labels, colors, width=1, font_size=12, fill=False):
    draw = ImageDraw.Draw(image)
    
    for box, label, color in zip(boxes, labels, colors):
        # Draw bounding box
        draw.rectangle(xy=box.tolist(), outline=color, width=width, fill=color if fill else None)
        
        # Set font size
        font = ImageFont.load_default()  # Load default font
        font = ImageFont.truetype("arial.ttf", font_size)  # Set font size
        
        # Draw label
        draw.text((box[0], box[1]), label, fill=color, font=font)
        
    return np.array(image)

def object_detection_yolo8(model = None, camera = 0, threshold = 0.5, font_size = 18):
    '''
    The purpose of this function is to connect to a webcam or external camera
    such as a GoPro and do object recognition using the YOLO v8 model. 
    
    Parameters
    ----------
    camera : Int, optional
        The type of camera. 0 is the built in webcam, 1-n is for other webcams.
        The default is 0.
        
    threshold : float, optional
        Value between 0 and 1. This is the confidence needed for the detection
        algorithm to show the predicted object. The default is 0.5.
        
    font_size : float, optional
        The font size of the titles for the predicted objects. The default is 
        18.

    '''
    # Load YOLOv8 model
    if model is None:
        model = YOLO('yolov8n.pt')
    # Open video capture
    cap = cv2.VideoCapture(camera)
    
    if not cap.isOpened():
        print("Cannot open camera. Aborting.")
        exit()
    else:
        print("Camera opened. Attempting to retrieve frame.")
    
    frame_counter = 0
    while True:
        ret, frame = cap.read()
    
        if not ret:
            print("Error retrieving frame. Aborting.")
            break
    
        results = model(frame, conf = threshold) # Perform object detection with YOLOv8
        # Extract bounding boxes, labels, and scores
        boxes = results[0].boxes.xyxy[:, :4].cpu().numpy()
        labels = [results[0].names[idx] for idx in results[0].boxes.data[:, 5].cpu().numpy().astype(int)]
        scores = results[0].boxes.conf[:, ].cpu().numpy()
    
        labels_with_scores = [f'{labels[i]} - {scores[i]:.2f}' for i in range(len(labels))]# if mask[i]]
    
        # Convert image to PIL format
        image_pil = Image.fromarray(frame)
    
        # Draw bounding boxes on image
        try:
            output = draw_bounding_boxes(image_pil, 
                                            boxes, 
                                            labels_with_scores, 
                                            ["red" if label=="person" else "green" for label in labels],
                                            font_size = font_size)
        except: output = np.array(image_pil)
    
        #Display the processed frame
        cv2.imshow('object detection using YOLO', output)
        frame_counter += 1
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

model = {'Nano': 'yolov8n.pt',
          'Small': 'yolov8s.pt',
          'Medium': 'yolov8m.pt',
          'Large': 'yolov8l.pt',
          'XLarge': 'yolov8x.pt'}

model = YOLO(model['Nano'])
object_detection_yolo8(model = model)    
    
