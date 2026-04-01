import numpy as np
import cv2
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad import TinyJit, Tensor
from blazeface import BlazeFace


def save_detections(original_img, detections, output_path="output.jpg"):
    if detections.ndim == 1: detections = np.expand_dims(detections, axis=0)

    img_out = original_img.copy()
    orig_h, orig_w = original_img.shape[:2]

    print("Found %d faces" % detections.shape[0])

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

if __name__ == '__main__':
    model = BlazeFace()
    orig = cv2.imread("nms.jpg")
    img = Tensor(orig)
    detections = model(img).numpy()
    detections = detections[detections[:, 4] != 0]

    expected = [[69.23135,66.04759,133.58755,130.40376,596.1233,],
[65.73704,239.54918,124.27883,298.09097,578.57294,],
[69.06696,372.31207,131.6906,434.93576,552.90955,],]

    np.testing.assert_allclose(detections, expected, rtol=1e-6, atol=1e-6)
    save_detections(original_img=orig, detections=detections, output_path="result.jpg")


