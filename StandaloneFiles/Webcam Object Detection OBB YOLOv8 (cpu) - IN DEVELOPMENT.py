import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def draw_outline_obb(image, boxes, labels, colors,width = 2, fill = False):
    draw = ImageDraw.Draw(image, "RGBA")
    #obb = result.obb
    for box, label, color in zip(boxes, labels, colors):
        # Draw bounding box
        draw.polygon(xy=box, outline=color, width=width, fill=color if fill else None)
        
        # Set font size
        font = ImageFont.load_default()  # Load default font
        font = ImageFont.truetype("arial.ttf", 18)  # Set font size
        
        # Draw label
        draw.text((box[0,0], box[0,1]), label, fill=color, font=font)
        
    return np.array(image)

#def object_pose_yolo8(camera = 0, threshold = 0.5, font_size = 18):
'''
The purpose of this function is to connect to a webcam or external camera
such as a GoPro and do object segmentation using the YOLO v8 model. 

Parameters
----------
camera : Int, optional
    The type of camera. 0 is the built in webcam, 1-n is for other webcams.
    The default is 0.

'''

    
model = YOLO('yolov8n-obb.pt')

# Open video capture
cap = cv2.VideoCapture(0)

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
    
    result = model.predict(frame)[0]
    image_pil = Image.fromarray(frame)
    boxes = result.obb.xyxyxyxy[:, :4].cpu().numpy()
    labels = [result.names[idx] for idx in result.obb.data[:, 5].cpu().numpy().astype(int)]
    scores = result.obb.conf[:, ].cpu().numpy()
    output = draw_outline_obb(image_pil, 
                                        boxes, 
                                        labels, 
                                        ["red" if label=="person" else "green" for label in labels],
                                        )
    #Display the processed frame
    
    cv2.imshow('object detection using YOLO', output)
    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

# #object_pose_yolo8()    
