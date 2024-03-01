import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def draw_lines(image, result, shape, width = 2):
    draw = ImageDraw.Draw(image)
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
    kpts = result.keypoints.xy[0]
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


def object_pose_yolo8(camera = 0, threshold = 0.5, font_size = 18):
    '''
    The purpose of this function is to connect to a webcam or external camera
    such as a GoPro and do object segmentation using the YOLO v8 model. 
    
    Parameters
    ----------
    camera : Int, optional
        The type of camera. 0 is the built in webcam, 1-n is for other webcams.
        The default is 0.
    
    '''
    # Load YOLOv8 model
    model = YOLO('yolov8n-pose.pt')
    
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
        shape = frame.shape[0:2]
        if not ret:
            print("Error retrieving frame. Aborting.")
            break
        result = model.predict(frame, conf=threshold)[0]
        image_pil = Image.fromarray(frame)
        output = draw_lines(image_pil, result, shape)
        #Display the processed frame
        
        cv2.imshow('Pose using YOLO', output)
        frame_counter += 1
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

object_pose_yolo8()    
