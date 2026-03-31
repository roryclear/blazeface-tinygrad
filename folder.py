import numpy as np
import cv2
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad import TinyJit, Tensor
from blazeface import BlazeFace
import random

import cv2
import numpy as np

def save_detections(original_img, detections, output_path="output.jpg", face_size=50, final_size=1000):
    if detections.ndim == 1: 
        detections = np.expand_dims(detections, axis=0)
    if detections.shape[0] == 0: 
        return
    
    det = detections[0]

    # First face bounding box
    ymin, xmin, ymax, xmax = det[:4]

    # Keypoints (6 points)
    keypoints = det[4:16].reshape(-1, 2)

    # ---- EYE MIDPOINT (blue + green = first two points) ----
    eye_mid_x = (keypoints[0][0] + keypoints[1][0]) / 2
    eye_mid_y = (keypoints[0][1] + keypoints[1][1]) / 2

    # Face center and dimensions (still used for scaling)
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

    # ---- USE EYE MIDPOINT INSTEAD OF BOX CENTER ----
    scaled_eye_x = eye_mid_x * scale_factor
    scaled_eye_y = eye_mid_y * scale_factor

    dx = final_size // 2 - scaled_eye_x
    dy = final_size // 2 - scaled_eye_y

    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    shifted_img = cv2.warpAffine(
        scaled_img,
        translation_matrix,
        (final_size, final_size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # ---- DRAW KEYPOINTS ----
    colors = [
        (255, 0, 0),    
        (0, 255, 0),    
        (0, 0, 255),    
        (255, 255, 0),  
        (255, 0, 255),  
        (0, 255, 255)   
    ]
    '''
    for i, (x, y) in enumerate(keypoints):
        sx = int(x * scale_factor + dx)
        sy = int(y * scale_factor + dy)
        cv2.circle(shifted_img, (sx, sy), 3, colors[i % len(colors)], -1)
    '''
    
    cv2.imwrite(f"face_imgs/{output_path}", shifted_img)
    
    # ---- FACE CROP (unchanged) ----
    new_width = face_width * 1.5
    new_height = face_height * 1.5
    
    x1 = int(face_center_x - new_width / 2)
    y1 = int(face_center_y - new_height / 2)
    x2 = int(face_center_x + new_width / 2)
    y2 = int(face_center_y + new_height / 2)
    
    x1 = max(0, min(orig_w, x1))
    x2 = max(0, min(orig_w, x2))
    y1 = max(0, min(orig_h, y1))
    y2 = max(0, min(orig_h, y2))
    
    face_box = original_img[y1:y2, x1:x2]
    
    if face_box.size > 0:
        face_box_resized = cv2.resize(face_box, (112, 112), interpolation=cv2.INTER_LINEAR)        
        cv2.imwrite(f"faces/{output_path}", face_box_resized)
    else:
        print("Warning: Face box extraction failed - invalid region")

@TinyJit
def jit_call(model, x): return model(x)

import os

def sort_detections_by_landmark_proximity(files_dets):
    print(len(files_dets))
    if len(files_dets) <= 1:
        return files_dets

    # Extract detections only
    detections_list = [d for d, _ in files_dets]

    # Stack into array for processing
    detections = np.array([d[0] for d in detections_list])

    # --- original logic (unchanged) ---
    # slop, all one for now
    # Index 0: left eye (green) - medium weight
    # Index 1: right eye (blue) - medium weight  
    # Index 2: nose (red) - high weight (good for orientation)
    # Index 3: right ear? (pink) - very high weight (excellent for orientation)
    # Index 4: left ear? (yellow?) - very high weight
    # Index 5: mouth (cyan) - medium weight

    landmark_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    all_keypoints = detections[:, 4:16].reshape(len(detections), -1, 2)

    relative_landmarks = []
    for keypoints in all_keypoints:
        eye_mid = (keypoints[0] + keypoints[1]) / 2
        rel_points = keypoints - eye_mid
        weighted_rel = rel_points * landmark_weights.reshape(-1, 1)
        relative_landmarks.append(weighted_rel)

    used = {0}
    order = [0]
    current = 0

    while len(order) < len(detections):
        best_idx, best_dist = None, float('inf')
        for i in range(len(detections)):
            if i in used:
                continue
            diff = relative_landmarks[current] - relative_landmarks[i]
            dist = np.sum(diff ** 2)
            if dist < best_dist:
                best_dist, best_idx = dist, i
        order.append(best_idx)
        used.add(best_idx)
        current = best_idx

    # Reorder original pairs
    return [files_dets[i] for i in order]

if __name__ == '__main__':
    model = BlazeFace()
    base_dir = "."  # or wherever your folders are

    files = []
    for folder in os.listdir(base_dir):
        if folder.startswith("objects_") and os.path.isdir(os.path.join(base_dir, folder)):
            folder_path = os.path.join(base_dir, folder)
            for file in os.listdir(folder_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    files.append(os.path.join(folder, file))

    files_dets = []

    for file in files:
        print(file)
        orig = cv2.imread(file)

        h, w = orig.shape[:2]
        scale = 640 / max(h, w)
        resized = cv2.resize(orig, (int(w*scale), int(h*scale)))

        delta_w, delta_h = 640 - resized.shape[1], 640 - resized.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0,0,0]
        )

        img = Tensor(padded)
        detections = jit_call(model, img).numpy()
        detections = detections[detections[:, 4] != 0]

        if len(detections) > 0: files_dets.append([detections, file])
    
    files_dets = sort_detections_by_landmark_proximity(files_dets)
    
    
    for i, (detections, file) in enumerate(files_dets):
        orig = cv2.imread(file)
        h, w = orig.shape[:2]
        scale = 640 / max(h, w)
        resized = cv2.resize(orig, (int(w*scale), int(h*scale)))

        delta_w, delta_h = 640 - resized.shape[1], 640 - resized.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0,0,0]
        )

        save_detections(
            original_img=padded,
            detections=detections,
            output_path=f"{i:04d}.jpg"
        )




'''
ffmpeg -framerate 24 -i %04d.jpg \
-vf "scale=1000:1000:force_original_aspect_ratio=decrease,pad=1000:1000:(ow-iw)/2:(oh-ih)/2" \
-c:v libx264 -pix_fmt yuv420p output.mp4
'''

#reverse
'''
ffmpeg -framerate 30 -i %04d.jpg \
-vf "reverse,scale=1000:1000:force_original_aspect_ratio=decrease,pad=1000:1000:(ow-iw)/2:(oh-ih)/2" \
-c:v libx264 -pix_fmt yuv420p output30.mp4
'''