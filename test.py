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
    state_dict = safe_load("model.safetensors")

    model = BlazeFace()
    load_state_dict(model, state_dict)


    model.min_score_thresh = 0.75

    orig = cv2.imread("messi.webp")
    img = Tensor(orig)
    detections = model(img).numpy()
    detections = detections[detections[:, 4] != 0]

    expected = [[196.61635,442.3443,276.1075,521.83545,588.39246,],[140.53587,233.816,226.6048,319.88492,539.11383,], ]

    #expected = [[0.22293027,0.3687327,0.35492355,0.500726, 0.4048541,0.253551,0.45936358,0.25396332,0.42835188,0.2809909,0.42859644,0.31245646,0.37655264,0.27385083,0.49636966,0.27672035,0.83855903,],
    #[0.30805102,0.68929595,0.42866126,0.8099063,0.71050656,0.34094658,0.75901216,0.34136337,0.7211923,0.3699867,0.7258061,0.3949228,0.703986,0.3506133,0.8086657,0.3542543,0.7997207,],]

    np.testing.assert_allclose(detections, expected, rtol=1e-6, atol=1e-6)

    save_detections(original_img=orig, detections=detections, output_path="result.jpg")


