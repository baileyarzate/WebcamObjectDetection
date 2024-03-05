import cv2
import numpy as np
import time
import keyboard
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssdlite320_mobilenet_v3_large
from pycocotools.coco import COCO
#from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torch
from PIL import ImageDraw, ImageFont
import numpy

def draw_bounding_boxes_n(image, boxes, labels, colors, width=1, font_size=12, fill=False):
    if isinstance(image, torch.Tensor):
        image = to_pil_image(image)
        
    draw = ImageDraw.Draw(image)
    
    for box, label, color in zip(boxes, labels, colors):
        # Draw bounding box
        draw.rectangle(xy=box.tolist(), outline=color, width=width, fill=color if fill else None)
        
        # Set font size
        font = ImageFont.load_default()  # Load default font
        font = ImageFont.truetype("arial.ttf", font_size)  # Set font size
        # Draw label
        draw.text((box[0], box[1]), label, fill=color, font=font)
        
    return image

#object_detection_model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
object_detection_model = fasterrcnn_resnet50_fpn(pretrained = True, progress = False)
object_detection_model.eval(); ## Setting Model for Evaluation/Prediction
annFile = r"C:\Users\baile\Downloads\annotations_trainval2017\annotations\instances_val2017.json"
coco = COCO(annFile)

cap = cv2.VideoCapture(0)

# If we failed to open, please let me know so I don't waste my time
if not cap.isOpened():
    print("Cannot open camera. Aborting.")
    exit()
else:
    print("Camera opened. Attempting to retrieve frame.")

frame_counter = 0
while True:
    # Constantly update the frame with the most up-to-date capture from the webcam
    ret, frame = cap.read()

    # Report failure or success to retrieve frame
    if not ret:
        print("Error retrieving frame. Aborting.")
        break
    else:
        if frame_counter == 0:
            print("Frame retrieved. Attempting to show.")
    
    # Show the raw webcam feed, for some reason it takes a while to get here (maybe 30 seconds?). Seems to have no issues once running
    #cv2.imshow("Raw", frame)
    if frame_counter == 0:
        print("Success!")
     
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    #_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw bounding boxes around each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the image with bounding boxes
    #cv2.imshow('Objects with Bounding Boxes', frame)
    #image = Image.open(r"C:\Users\baile\Pictures\IMG_6571.jpg").rotate(-90)
    #image = frame
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_tensor_int = pil_to_tensor(image).unsqueeze(dim=0)/255.0
    image_preds = object_detection_model(image_tensor_int)
    image_labels = coco.loadCats(image_preds[0]["labels"].numpy())
    image_annot_labels = ["{}-{:.2f}".format(label["name"], prob) for label, prob in zip(image_labels, image_preds[0]["scores"].detach().numpy())]
    image_array = image_tensor_int[0].detach().cpu().numpy()  # Convert to numpy array
    image_array = (image_array * 255).astype(np.uint8)  # Convert to uint8
    image_tensor = torch.from_numpy(image_array)  # Convert back to Torch tensor
    output = draw_bounding_boxes_n(image=image_tensor,
                                        boxes=image_preds[0]["boxes"],
                                        labels=image_annot_labels,
                                        colors=["red" if label["name"]=="person" else "green" for label in image_labels],
                                        width=5,
                                        font_size=30,
                                        fill=False
                                        )
    #cv2.imshow('object detection using NN', output)
    output = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
    cv2.imshow('object detection using NN', output) #**
    #output.show()
    # #Show the BGR version
    # bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imshow('bgr', bgr)
    
    # # Show the Laplacian of the raw feed
    # laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    # laplacian = np.uint8(laplacian)
    # cv2.imshow('Laplacian', laplacian)
    
    # Show the Canny edge detection output
    #edges = cv2.Canny(frame, 100, 100)
    #cv2.imshow('Canny', edges)
    
    # Start logging time for frame rate analysis
    if frame_counter == 0:
        start_time = time.time()
    frame_counter += 1

    # Calculate average frame rate upon hitting the X frame.
    if frame_counter == 100:
        frame_rate = 100 / (time.time() - start_time)
        print("Average FPS for the first 1000 frames: %s" % frame_rate)

    # Abort with 'q' or whatever you want to use
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Let other people use the webcam when we are done    
cap.release()
cv2.destroyAllWindows()

