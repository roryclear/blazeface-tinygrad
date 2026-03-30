import numpy as np
import cv2
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad import TinyJit, Tensor
from blazeface import BlazeFace
import random

def save_detections(original_img, detections, output_path="output.jpg", face_size=50, final_size=1000):
    if detections.ndim == 1: 
        detections = np.expand_dims(detections, axis=0)
    if detections.shape[0] == 0: return

    # First face bounding box
    ymin, xmin, ymax, xmax = detections[0, :4]

    # Face center and dimensions
    face_center_x = (xmin + xmax) / 2
    face_center_y = (ymin + ymax) / 2
    face_width = xmax - xmin
    face_height = ymax - ymin

    # Scale factor to make face 50x50
    scale_factor = face_size / max(face_width, face_height)

    # Scale original image
    orig_h, orig_w = original_img.shape[:2]
    scaled_w = int(orig_w * scale_factor)
    scaled_h = int(orig_h * scale_factor)
    scaled_img = cv2.resize(original_img, (scaled_w, scaled_h))

    # Face center in scaled image
    scaled_face_center_x = face_center_x * scale_factor
    scaled_face_center_y = face_center_y * scale_factor


    # Translation to center face in final canvas
    dx = final_size // 2 - scaled_face_center_x
    dy = final_size // 2 - scaled_face_center_y
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_img = cv2.warpAffine(scaled_img, translation_matrix, (final_size, final_size),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # Draw bounding boxes
    '''
    for i in range(detections.shape[0]):
        box_ymin = detections[i,0]*scale_factor + dy
        box_xmin = detections[i,1]*scale_factor + dx
        box_ymax = detections[i,2]*scale_factor + dy
        box_xmax = detections[i,3]*scale_factor + dx

        x1 = max(0, min(final_size, int(box_xmin)))
        x2 = max(0, min(final_size, int(box_xmax)))
        y1 = max(0, min(final_size, int(box_ymin)))
        y2 = max(0, min(final_size, int(box_ymax)))

        color = (0,0,255) if i == 0 else (0,255,0)
        cv2.rectangle(shifted_img, (x1,y1), (x2,y2), color, 2)
    '''
    #cv2.imwrite(f"faces/{str(random.randint(1, 1000000000))}.jpg", shifted_img)
    cv2.imwrite(output_path, shifted_img)

@TinyJit
def jit_call(model, x): return model(x)

import os

if __name__ == '__main__':
    model = BlazeFace()

    files = os.listdir("objects")
    
    for file in files:
        print(file)
        orig = cv2.imread(f"objects/{file}")

        h, w = orig.shape[:2]
        scale = 640 / max(h, w)
        resized = cv2.resize(orig, (int(w*scale), int(h*scale)))
        delta_w, delta_h = 640 - resized.shape[1], 640 - resized.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        orig = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

        img = Tensor(orig)
        detections = jit_call(model, img).numpy()
        detections = detections[detections[:, 4] != 0]

        save_detections(original_img=orig, detections=detections, output_path=f"faces/{file}")


