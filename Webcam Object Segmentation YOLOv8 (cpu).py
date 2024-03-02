import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def draw_outline(image, result, width = 2):
    draw = ImageDraw.Draw(image, "RGBA")
    masks = result.masks
    for i in range(len(masks)):
        draw.polygon(masks[i].xy[0],outline=(0,255,0), 
                     width=width,
                     fill = (0,255,0,50))
    return np.array(image)

def object_segmentation_yolo8(model = None, camera = 0, threshold = 0.5, font_size = 18):
    '''
    The purpose of this function is to connect to a webcam or external camera
    such as a GoPro and do object segmentation using the YOLO v8 model. 
    
    Parameters
    ----------
    camera : Int, optional
        The type of camera. 0 is the built in webcam, 1-n is for other webcams.
        The default is 0.

    '''
    # Open video capture
    cap = cv2.VideoCapture(camera)
    if model is None:
        model = YOLO('yolov8n-seg.pt')
        
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
    
        result = model(frame)[0]
        image_pil = Image.fromarray(frame)
        try: output = draw_outline(image_pil, result)
        except: output = np.array(image_pil)
        #Display the processed frame
        
        cv2.imshow('object segmentation using YOLO', output)
        frame_counter += 1
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#model sizes:
#   - Nano: yolov8n-seg.pt
#   - Small: yolov8s-seg.pt
#   - Medium: yolov8m-seg.pt
#   - Large: yolov8l-seg.pt
#   - XLarge: yolov8x-seg.pt

#Load in the model
model = YOLO('yolov8n-seg.pt')
#Run
object_segmentation_yolo8(model = model)    