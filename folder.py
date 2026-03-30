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

    cv2.imwrite(f"face_imgs/{output_path}", shifted_img)
    
    # Save double-sized face box as 112x112
    # Calculate double-sized bounding box (keep same center)
    new_width = face_width * 1.5
    new_height = face_height * 1.5
    
    # Calculate coordinates in original image
    x1 = int(face_center_x - new_width / 2)
    y1 = int(face_center_y - new_height / 2)
    x2 = int(face_center_x + new_width / 2)
    y2 = int(face_center_y + new_height / 2)
    
    # Clamp to image boundaries
    x1 = max(0, min(orig_w, x1))
    x2 = max(0, min(orig_w, x2))
    y1 = max(0, min(orig_h, y1))
    y2 = max(0, min(orig_h, y2))
    
    # Extract face region
    face_box = original_img[y1:y2, x1:x2]
    
    # Resize to 112x112
    if face_box.size > 0:
        face_box_resized = cv2.resize(face_box, (112, 112), interpolation=cv2.INTER_LINEAR)        
        cv2.imwrite(f"faces/{output_path}", face_box_resized)
    else:
        print("Warning: Face box extraction failed - invalid region")

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

        save_detections(original_img=orig, detections=detections, output_path=f"{file}")


