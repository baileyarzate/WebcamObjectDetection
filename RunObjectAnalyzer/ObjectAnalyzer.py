import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

class ObjectAnalyzer:
    def __init__(self, model = 'yolov8n-seg.pt', conf = 0.5, bbox = True): 
        if model[8:11] == 'obb':
            print("WARNING: Oriented Bounded Box may not perform as expected!")
        self.model = YOLO(model)
        self.result = None
        self.texts = []
        self.conf = conf
        self.iterations = 0
        self.bbox = bbox
        pass
    
    def detect(self, image, width=1, font_size=12, fill=False):
        self.result = self.model.predict(image, conf = self.conf)[0]
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        boxes = self.result.boxes.xyxy[:, :4].cpu().numpy()
        labels = [self.result.names[idx] for idx in self.result.boxes.data[:, 5].cpu().numpy().astype(int)]
        scores = self.result.boxes.conf[:, ].cpu().numpy()
        colors = ["red" if label=="person" else "green" for label in labels]
        labels = [f'{labels[i]} - {scores[i]:.2f}' for i in range(len(labels))]
        for box, label, color in zip(boxes, labels, colors):
            # Draw bounding box
            draw.rectangle(xy=box.tolist(), outline=color, width=width, fill=color if fill else None)
            
            # Set font size
            font = ImageFont.load_default()  # Load default font
            font = ImageFont.truetype("arial.ttf", font_size)  # Set font size
            
            # Draw label
            draw.text((box[0], box[1]), label, fill=color, font=font)
            
        return np.array(image)
        
    def pose(self, image, width = 2):
        shape = image.shape[0:2]
        self.result = self.model.predict(image, conf = self.conf)[0]
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image, "RGBA")
        if self.bbox: #If you want to show bounding boxes make self.bbox = True
            boxes = self.result.boxes.xyxy[:, :4].cpu().numpy()
            labels = [self.result.names[idx] for idx in self.result.boxes.data[:, 5].cpu().numpy().astype(int)]
            scores = self.result.boxes.conf[:, ].cpu().numpy()
            colors = ["red" if label=="person" else "green" for label in labels]
            labels = [f'{labels[i]} - {scores[i]:.2f}' for i in range(len(labels))]
            for box, label, color in zip(boxes, labels, colors):
                # Draw bounding box
                draw.rectangle(xy=box.tolist(), outline=color, width=width, fill=(0,255,0,50))
                
                # Set font size
                font = ImageFont.load_default()  # Load default font
                font = ImageFont.truetype("arial.ttf", 18)  # Set font size
                
                # Draw label
                draw.text((box[0], box[1]), label, fill=color, font=font)
        palette = np.array(
                    [[255, 128, 0],[255, 153, 51],[255, 178, 102],[230, 230, 0],
                     [255, 153, 255],[153, 204, 255],[255, 102, 255],[255, 51, 255],
                     [102, 178, 255],[51, 153, 255],[255, 153, 153],[255, 102, 102],
                     [255, 51, 51],[153, 255, 153],[102, 255, 102],[51, 255, 51],
                     [0, 255, 0],[0, 0, 255],[255, 0, 0],[255, 255, 255],],
                    dtype=np.uint8,)

        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        rad = 2
        steps = 1
        kpts = self.result.keypoints.xy[0]
        #num_kpts = len(kpts) // steps # may be useful at another time

        for i, k in enumerate(kpts):
            r, g, b = pose_kpt_color[i]
            fillcolor = (int(r), int(g), int(b))
            x, y = k[0], k[1]
            if x % shape[1] != 0 and y % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue    
                draw.ellipse((x-rad, y-rad,x+rad,y+rad), fill = fillcolor, width=3)  

        for sk_id, sk in enumerate(skeleton):
              r, g, b = pose_limb_color[sk_id]
              fillcolor = (int(r), int(g), int(b))
              pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
              pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
              if steps == 3:
                  conf1 = kpts[(sk[0] - 1), 2]
                  conf2 = kpts[(sk[1] - 1), 2]
                  if conf1<0.5 or conf2<0.5:
                      continue
              if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                  continue
              if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                  continue
              draw.line([pos1, pos2], fillcolor, width=2)
        return np.array(image)
    
    def segment(self, image):
        self.result = self.model.predict(image, conf = self.conf)[0]
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image, "RGBA")
        masks = self.result.masks
        for i in range(len(masks)):
            draw.polygon(masks[i].xy[0],outline=(0,255,0), 
                         width=2,
                         fill = (0,255,0,50))
        return np.array(image)
    
    def classify(self, image, result = None):
        self.result = self.model(image, conf = self.conf)[0]
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image, "RGBA")
        box_width = 200  # You can adjust the width as needed
        box_height = 100  # You can adjust the height as needed
        spacing = 20  # Spacing between boxes
        start_x = 50  # Starting x-coordinate for the boxes
        font = ImageFont.truetype("arial.ttf", 16)
        # Draw stacked boxes
        for i, text in enumerate(self.texts):
            box = (start_x, 50 + i * (box_height + spacing), start_x + box_width, 50 + i * (box_height + spacing) + box_height)
            draw.rectangle(box, outline="black", fill= (0,0,0,125))
            draw.text((start_x + 10, 50 + i * (box_height + spacing) + 10), text, fill="white", font = font)

        # Define text for each box
        if self.iterations % 200 == 0:
            self.texts = []
            if len(self.texts) == 0:
                for i in range(3):
                    self.texts.append(self.result.names[self.result.probs.top5[i]])   
        self.iterations += 1
        return np.array(image)
    
    def obb(self, image, width = 2, fill = False):
        self.result = self.model.predict(image)[0]
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image, "RGBA")
        boxes = self.result.obb.xyxyxyxy[:, :4].cpu().numpy()
        labels = [self.result.names[idx] for idx in self.result.obb.data[:, 5].cpu().numpy().astype(int)]
        scores = self.result.obb.conf[:, ].cpu().numpy()
        colors = ["red" if label=="person" else "green" for label in labels]
        labels = [f'{labels[i]} - {scores[i]:.2f}' for i in range(len(labels))]
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
    
def objectAnalyzerRun(model, cameraType = 0, conf = 0.5, file_path = None, mediaType = 'webcam'):
    Obj1 = ObjectAnalyzer(model = model, conf = 0.5) 
    if model[8:11] == 'seg':
        title = 'Object Detection using YOLO'
        modelType = 'Segmentation'
    elif model[8:12] == 'pose':
        title = 'Pose using YOLO'
        modelType = 'Pose'
    elif model[8:11] == 'cls':
        title = 'Object Classification using YOLO'
        modelType = 'Classification'
    elif model[8:11] == 'obb':
        title = 'OBB Detection using YOLO'
        modelType = 'Oriented Bounding Box'
    else:
        title = 'Object Detection using YOLO'
        modelType = 'Detection'

    if mediaType == 'webcam' or mediaType == 'video':
        if mediaType == 'webcam':
            cap = cv2.VideoCapture(cameraType)
        elif mediaType == 'video':
            cap = cv2.VideoCapture(file_path)
    
        while True:
            ret, frame = cap.read()
            if mediaType == 'video':
                frame = cv2.resize(frame, (640, 480))
            if not ret:
                print("Error retrieving frame. Aborting.")
                break
            try:
                if modelType == 'Segmentation':
                    output = Obj1.segment(frame)
                elif modelType == "Pose":
                    output = Obj1.pose(frame)
                elif modelType == 'Detection':
                    output = Obj1.detect(frame)
                elif modelType == 'Classification':
                    output = Obj1.classify(frame)
                elif modelType == 'Oriented Bounding Box': 
                    output = Obj1.obb(frame)
            except: output = frame
            
            #Display the processed frame
            cv2.imshow(title, output)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
    elif mediaType == 'picture':
        frame = cv2.imread(file_path)
        frame = cv2.resize(frame, (640, 480))
        try:
            if modelType == 'Segmentation':
                output = Obj1.segment(frame)
            elif modelType == "Pose":
                output = Obj1.pose(frame)
            elif modelType == 'Detection':
                output = Obj1.detect(frame)
            elif modelType == 'Classification':
                output = Obj1.classify(frame)
            elif modelType == 'Oriented Bounding Box': 
                output = Obj1.obb(frame)
        except:
            output = frame
        
        cv2.imshow(title, output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#TASKS
#1. Make this work for pictures (DONE)
#2. Make this work for pre-recorded videos (DONE)
#3. Make a GUI so the opertor has complete control without having to code. 


#models (nano):
#   - Detection: yolov8n.pt
#   - Classification: yolov8n-cls.pt
#   - Pose: yolov8n-pose.pt
#   - Segmentation: yolov8n-seg.pt
#   - Oriented Bounding Box: yolov8n-obb.pt

# Each model has larger sizes available:
    # yolov8s (small), yolov8m (medium), yolov8l (large), yolov8sx (Xlarge)
    
# Example usage:
    # objectAnalyzerRun(model = yolov8l-seg.pt)

#objectAnalyzerRun(model = 'yolov8n.pt', cameraType = 0, mediaType = 'webcam', conf = 0.5)