import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch import Tensor
from tinygrad import Tensor as tinyTensor, nn as tiny_nn
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict


class BlazeBlock_tiny():
    def __init__(self, c=None, channel_pad=0):
        if c is not None:
            self.stride = c.stride
            self.channel_pad = c.channel_pad
            self.conv0_tiny = c.conv0_tiny
            self.conv1_tiny = c.conv1_tiny
            return
        
        self.channel_pad = channel_pad
    
    def __call__(self, x):
        if self.stride == 2:
            h = x.pad(((0, 0), (0, 0), (0, 2), (0, 2)))
            x = x.max_pool2d(self.stride, self.stride)
        else:
            h = x

        if self.channel_pad > 0:
            x = x.pad(((0, 0), (0, self.channel_pad), (0, 0), (0, 0)))


        h = self.conv0_tiny(h)
        h = self.conv1_tiny(h)
        x += h
        x = x.relu()
        return x

def to_tiny(x): return tinyTensor(x.detach().numpy())

def to_torch(x): return Tensor(x.numpy())

def to_tiny_seq(x):
    ret = tiny_Seq(size=len(x))
    for i in range(len(x)): ret.list[i] = x[i]
    return ret

class tiny_Seq():
    def __init__(self, size=0):
        super().__init__()
        self.list = [None] * size
    def __len__(self): return len(self.list)
    def __setitem__(self, key, value): self.list[key] = value
    def __getitem__(self, idx): return self.list[idx]
    def __call__(self, x):
        for y in self.list: x = y(x)
        return x

class FinalBlazeBlock_tiny():
    def __init__(self, f=None):
        if f is not None:
            self.act = f.act
            self.convs = f.convs
            self.conv0_tiny = f.conv0_tiny
            self.conv1_tiny = f.conv1_tiny
            return
        
        self.conv0_tiny = tiny_nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0, groups=96, bias=True)
        self.conv1_tiny = tiny_nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0, bias=True)

    def __call__(self, x):
        x = x.pad(((0, 0), (0, 0), (0, 2), (0, 2)))
        x = self.conv0_tiny(x)
        x = self.conv1_tiny(x)
        x = x.relu()
        return x

class BlazeFace_tiny():
    def __init__(self, m=None):
        if m is not None:
            self.backbone_tiny = m.backbone_tiny
            self.conv_tiny = m.conv_tiny
            self.classifier_8 = m.classifier_8
            self.classifier_16 = m.classifier_16
            self.regressor_8 = m.regressor_8
            self.regressor_16 = m.regressor_16
            self.anchors = m.anchors
            self.x_scale = m.x_scale
            self.y_scale = m.y_scale
            self.w_scale = m.w_scale
            self.h_scale = m.h_scale
            self.score_clipping_thresh = m.score_clipping_thresh
            self.min_score_thresh = m.min_score_thresh
            self.min_suppression_threshold = m.min_suppression_threshold
            self.final = m.final
            return
        
        self.conv_tiny = tiny_nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True)
        self.classifier_8_tiny = tiny_nn.Conv2d(in_channels=96, out_channels=2, kernel_size=1, groups=1, bias=True)
        self.classifier_16_tiny = tiny_nn.Conv2d(in_channels=96, out_channels=6, kernel_size=1, groups=1, bias=True)
        self.regressor_8_tiny = tiny_nn.Conv2d(in_channels=96, out_channels=32, kernel_size=1, groups=1, bias=True)
        self.regressor_16_tiny = tiny_nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, groups=1, bias=True)

        self.final = FinalBlazeBlock_tiny()
        self.backbone_tiny = tiny_Seq(31)
        for i in range(7):
            self.backbone_tiny[i] = BlazeBlock_tiny()
            self.backbone_tiny[i].conv0_tiny = tiny_nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, groups=24, bias=True)
            self.backbone_tiny[i].conv1_tiny = tiny_nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
            self.backbone_tiny[i].stride = 1
        self.backbone_tiny[7] = BlazeBlock_tiny()
        self.backbone_tiny[7].conv0_tiny = tiny_nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=0, groups=24, bias=True)
        self.backbone_tiny[7].conv1_tiny = tiny_nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.backbone_tiny[7].stride = 2
        for i in range(8, 15):
            self.backbone_tiny[i] = BlazeBlock_tiny()
            self.backbone_tiny[i].conv0_tiny = tiny_nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, groups=24, bias=True)
            self.backbone_tiny[i].conv1_tiny = tiny_nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
            self.backbone_tiny[i].stride = 1
        self.backbone_tiny[15] = BlazeBlock_tiny(channel_pad=24)
        self.backbone_tiny[15].conv0_tiny = tiny_nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=0, groups=24, bias=True)
        self.backbone_tiny[15].conv1_tiny = tiny_nn.Conv2d(24, 48, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.backbone_tiny[15].stride = 2
        for i in range(16, 23):
            self.backbone_tiny[i] = BlazeBlock_tiny()
            self.backbone_tiny[i].conv0_tiny = tiny_nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=48, bias=True)
            self.backbone_tiny[i].conv1_tiny = tiny_nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
            self.backbone_tiny[i].stride = 1
        self.backbone_tiny[23] = BlazeBlock_tiny(channel_pad=48)
        self.backbone_tiny[23].conv0_tiny = tiny_nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=0, groups=48, bias=True)
        self.backbone_tiny[23].conv1_tiny = tiny_nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.backbone_tiny[23].stride = 2
        for i in range(24, 31):
            self.backbone_tiny[i] = BlazeBlock_tiny()
            self.backbone_tiny[i].conv0_tiny = tiny_nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96, bias=True)
            self.backbone_tiny[i].conv1_tiny = tiny_nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
            self.backbone_tiny[i].stride = 1

        self.anchors = tinyTensor.empty(896, 4)
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0

        self.x_scale = 256.0
        self.y_scale = 256.0
        self.h_scale = 256.0
        self.w_scale = 256.0
        self.min_score_thresh = 0.65
        self.min_suppression_threshold = 0.3


    def __call__(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = x.pad(((0, 0), (0, 0), (1, 2), (1, 2)))
        
        b = x.shape[0]      # batch size, needed for reshaping later
        x = self.conv_tiny(x)
        x = x.relu()
        x = self.backbone_tiny(x)           # (b, 16, 16, 96)

        h = self.final(x)              # (b, 8, 8, 96)

        # Note: Because PyTorch is NCHW but TFLite is NHWC, we need to
        # permute the output from the conv layers before reshaping it.
        
        c1 = self.classifier_8_tiny(x)       # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)     # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 1)       # (b, 512, 1)

        c2 = self.classifier_16_tiny(h)      # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)     # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 1)       # (b, 384, 1)

        c = tinyTensor.cat(c1, c2, dim=1)
        r1 = self.regressor_8_tiny(x)        # (b, 32, 16, 16)
        r1 = r1.permute(0, 2, 3, 1)     # (b, 16, 16, 32)
        r1 = r1.reshape(b, -1, 16)      # (b, 512, 16)
        
        r2 = self.regressor_16_tiny(h)       # (b, 96, 8, 8)
        r2 = r2.permute(0, 2, 3, 1)     # (b, 8, 8, 96)
        r2 = r2.reshape(b, -1, 16)      # (b, 384, 16)

        r = tinyTensor.cat(r1, r2, dim=1)
        return [r, c]



    def predict_on_image(self, x):
        x = tinyTensor(x)
        x = x.permute((2, 0, 1))
        x = x.unsqueeze(0)
        x = x / 127.5 - 1.0
        out = self.__call__(x)

        out[0] = to_torch(out[0])
        out[1] = to_torch(out[1])

        detections = self._tensors_to_detections(out[0], out[1], self.anchors)

        detections = tinyTensor.cat(detections[:, :4], detections[:, 16:17], dim=1)
        
        return postprocess(detections)[0]

    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)  # (B, N, 16)
        thresh = self.score_clipping_thresh
        raw_score_tensor = to_tiny(raw_score_tensor)
        scores = raw_score_tensor.clip(-thresh, thresh).sigmoid().squeeze(-1)
        mask = scores >= self.min_score_thresh  # (B, N)
        scores = scores.unsqueeze(-1)  # (B, N, 1)
        detections = tinyTensor.cat(detection_boxes, scores, dim=-1)  # (B, N, 17)
        detections *= mask.unsqueeze(-1)
        return detections[0]
    
    def _decode_boxes(self, raw_boxes, anchors):
        raw_boxes = to_tiny(raw_boxes)
        anchors = to_tiny(anchors)
        boxes = tinyTensor.zeros_like(raw_boxes).contiguous()
        ax = anchors[:, 0]
        ay = anchors[:, 1]
        aw = anchors[:, 2]
        ah = anchors[:, 3]
        x_center = raw_boxes[..., 0] / self.x_scale * aw + ax
        y_center = raw_boxes[..., 1] / self.y_scale * ah + ay
        w = raw_boxes[..., 2] / self.w_scale * aw
        h = raw_boxes[..., 3] / self.h_scale * ah
        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax
        keypoints = raw_boxes[..., 4:].view(*raw_boxes.shape[:-1], 6, 2)
        kp_x = keypoints[..., 0] / self.x_scale * aw.unsqueeze(0).unsqueeze(-1) + ax.unsqueeze(0).unsqueeze(-1)
        kp_y = keypoints[..., 1] / self.y_scale * ah.unsqueeze(0).unsqueeze(-1) + ay.unsqueeze(0).unsqueeze(-1)
        keypoints_decoded = tinyTensor.stack((kp_x, kp_y), dim=-1)  # (B, N, 6, 2)
        boxes[..., 4:] = keypoints_decoded.view(*raw_boxes.shape[:-1], -1)
        return boxes

def postprocess(boxes):
    boxes = boxes.unsqueeze(0)
    max_det = 896
    iou_threshold = 0.3
    conf_threshold = 0.75 
    probs = boxes[:, :, 4] 
    order_all = tinyTensor.topk(probs, min(max_det, probs.shape[1]))[1]
    batch_idx = tinyTensor.arange(order_all.shape[0]).reshape(-1, 1)
    boxes = boxes[batch_idx, order_all]
    ious = compute_iou_matrix(boxes[:, :, :4])
    ious = tinyTensor.triu(ious, diagonal=1)
    high_iou_mask = (ious > iou_threshold)
    no_overlap_mask = high_iou_mask.sum(axis=1) == 0
    conf_mask = (boxes[:, :, 4] >= conf_threshold)
    final_mask = no_overlap_mask * conf_mask
    return boxes * final_mask.unsqueeze(-1)

def compute_iou_matrix(boxes):
  x1s = boxes[:, :, 0]
  y1s = boxes[:, :, 1]
  x2s = boxes[:, :, 2]
  y2s = boxes[:, :, 3]
  areas = (x2s - x1s) * (y2s - y1s)
  x1 = tinyTensor.maximum(x1s[:, :, None], x1s[:, None, :])
  y1 = tinyTensor.maximum(y1s[:, :, None], y1s[:, None, :])
  x2 = tinyTensor.minimum(x2s[:, :, None], x2s[:, None, :])
  y2 = tinyTensor.minimum(y2s[:, :, None], y2s[:, None, :])
  w = tinyTensor.maximum(tinyTensor(0), x2 - x1)
  h = tinyTensor.maximum(tinyTensor(0), y2 - y1)
  intersection = w * h
  union = areas[:, :, None] + areas[:, None, :] - intersection
  return intersection / union

# IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py


def save_detections_on_original(
    original_img,
    detections,
    scale,
    pad_top,
    pad_left,
    resized_shape,
    output_path="output.jpg"
):
    if detections.ndim == 1: detections = np.expand_dims(detections, axis=0)

    img_out = original_img.copy()
    orig_h, orig_w = original_img.shape[:2]
    resized_h, resized_w = resized_shape

    print("Found %d faces" % detections.shape[0])

    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * resized_h
        xmin = detections[i, 1] * resized_w
        ymax = detections[i, 2] * resized_h
        xmax = detections[i, 3] * resized_w

        ymin -= pad_top
        ymax -= pad_top
        xmin -= pad_left
        xmax -= pad_left

        ymin /= scale
        ymax /= scale
        xmin /= scale
        xmax /= scale

        x1, y1, x2, y2 = map(int, [xmin, ymin, xmax, ymax])

        x1 = max(0, min(orig_w, x1))
        x2 = max(0, min(orig_w, x2))
        y1 = max(0, min(orig_h, y1))
        y2 = max(0, min(orig_h, y2))

        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(output_path, img_out)


state_dict = safe_load("model.safetensors")

model_tiny2 = BlazeFace_tiny()
load_state_dict(model_tiny2, state_dict)


model_tiny2.min_score_thresh = 0.75



orig = cv2.imread("messi.webp")
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

h0, w0 = orig.shape[:2]

scale = min(256 / w0, 256 / h0)
new_w, new_h = int(w0 * scale), int(h0 * scale)

resized = cv2.resize(orig, (new_w, new_h))

pad_top = (256 - new_h) // 2
pad_bottom = (256 - new_h) - pad_top
pad_left = (256 - new_w) // 2
pad_right = (256 - new_w) - pad_left

img = cv2.copyMakeBorder(
    resized,
    pad_top, pad_bottom, pad_left, pad_right,
    borderType=cv2.BORDER_CONSTANT,
    value=[0, 0, 0]
)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

detections = model_tiny2.predict_on_image(img).numpy()
detections = detections[detections[:, 4] != 0]
detections = detections[:, :4]

expected = [[0.22250968,0.36720884,0.35707378,0.501773,],[0.3098331,0.68761027,0.43034846,0.80812573,],]

#expected = [[0.22293027,0.3687327,0.35492355,0.500726, 0.4048541,0.253551,0.45936358,0.25396332,0.42835188,0.2809909,0.42859644,0.31245646,0.37655264,0.27385083,0.49636966,0.27672035,0.83855903,],
#[0.30805102,0.68929595,0.42866126,0.8099063,0.71050656,0.34094658,0.75901216,0.34136337,0.7211923,0.3699867,0.7258061,0.3949228,0.703986,0.3506133,0.8086657,0.3542543,0.7997207,],]

np.testing.assert_allclose(detections, expected, rtol=1e-6, atol=1e-6)

save_detections_on_original(
    original_img=cv2.cvtColor(orig, cv2.COLOR_RGB2BGR),
    detections=detections,
    scale=scale,
    pad_top=pad_top,
    pad_left=pad_left,
    resized_shape=(256, 256),
    output_path="result.jpg"
)


