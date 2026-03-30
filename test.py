import numpy as np
import cv2
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad import TinyJit, Tensor
from blazeface import BlazeFace


def save_detections(original_img, detections, output_path="output.jpg"):
    if detections.ndim == 1: detections = np.expand_dims(detections, axis=0)

    img_out = original_img.copy()
    orig_h, orig_w = original_img.shape[:2]

    if detections.shape[0] == 0: return

    for i in range(detections.shape[0]):
        ymin = detections[i, 0]
        xmin = detections[i, 1]
        ymax = detections[i, 2]
        xmax = detections[i, 3]

        x1, y1, x2, y2 = map(int, [xmin, ymin, xmax, ymax])

        x1 = max(0, min(orig_w, x1))
        x2 = max(0, min(orig_w, x2))
        y1 = max(0, min(orig_h, y1))
        y2 = max(0, min(orig_h, y2))

        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(output_path, img_out)

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

        save_detections(original_img=orig, detections=detections, output_path=f"faces/{file}.jpg")


