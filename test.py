import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch import Tensor
from tinygrad import Tensor as tinyTensor, nn as tiny_nn

#class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)[source]

#def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...], stride=1, padding:int|tuple[int, ...]|str=0, dilation=1, groups=1, bias=True):

class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # TFLite uses slightly different padding than PyTorch 
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, 
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.conv0_tiny = tiny_nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=True)
        self.conv1_tiny = tiny_nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = to_tiny(x)
        if self.stride == 2:
            h = x.pad(((0, 0), (0, 0), (0, 2), (0, 2)))
            x = x.max_pool2d(self.stride, self.stride)
        else:
            h = x

        if self.channel_pad > 0:
            x = x.pad(((0, 0), (0, self.channel_pad), (0, 0), (0, 0)))


        x = to_torch(x)
        h = to_torch(h)
        h = self.convs[0](h)
        h = self.convs[1](h)

        return self.act(h + x)

def to_tiny(x): return tinyTensor(x.detach().numpy())

def to_torch(x): return Tensor(x.numpy())

class FinalBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(FinalBlazeBlock, self).__init__()
                                                      # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, stride=2, padding=0,
                      groups=channels, bias=True),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = F.pad(x, (0, 2, 0, 2), "constant", 0)

        return self.act(self.convs(h))


class BlazeFace(nn.Module):
    def __init__(self):
        super(BlazeFace, self).__init__()

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

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24, stride=2),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 48, stride=2),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 96, stride=2),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
        )

        self.final = FinalBlazeBlock(96)
        self.classifier_8 = nn.Conv2d(96, 2, 1, bias=True)
        self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

        self.regressor_8 = nn.Conv2d(96, 32, 1, bias=True)
        self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        
        b = x.shape[0]      # batch size, needed for reshaping later

        x = self.backbone(x)           # (b, 16, 16, 96)
        h = self.final(x)              # (b, 8, 8, 96)
        
        # Note: Because PyTorch is NCHW but TFLite is NHWC, we need to
        # permute the output from the conv layers before reshaping it.
        
        c1 = self.classifier_8(x)       # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)     # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 1)       # (b, 512, 1)

        c2 = self.classifier_16(h)      # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)     # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 1)       # (b, 384, 1)

        c = torch.cat((c1, c2), dim=1)  # (b, 896, 1)

        r1 = self.regressor_8(x)        # (b, 32, 16, 16)
        r1 = r1.permute(0, 2, 3, 1)     # (b, 16, 16, 32)
        r1 = r1.reshape(b, -1, 16)      # (b, 512, 16)

        r2 = self.regressor_16(h)       # (b, 96, 8, 8)
        r2 = r2.permute(0, 2, 3, 1)     # (b, 8, 8, 96)
        r2 = r2.reshape(b, -1, 16)      # (b, 384, 16)

        r = torch.cat((r1, r2), dim=1)  # (b, 896, 16)
        return [r, c]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.classifier_8.weight.device
       

    def _preprocess(self, x): return x.float() / 127.5 - 1.0

    def predict_on_image(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))

        return self.predict_on_batch(img.unsqueeze(0))[0]

    def predict_on_batch(self, x):
        x = x.to(self._device())
        x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)

        # 3. Postprocess the raw predictions:
        detections = self._tensors_to_detections(out[0], out[1], self.anchors)

        # 4. Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, 17))
            filtered_detections.append(faces)

        return filtered_detections

    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)
        
        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
        detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)

        mask = detection_scores >= self.min_score_thresh

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        # todo, static!
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            output_detections.append(torch.cat((boxes, scores), dim=-1))

        return output_detections

    def _decode_boxes(self, raw_boxes, anchors):
        boxes = torch.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(6): # todo, vectorize?
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset    ] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _weighted_non_max_suppression(self, detections): # todo, vectorize nms
        if len(detections) == 0: return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = torch.argsort(detections[:, 16], descending=True)

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other 
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.clone()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :16]
                scores = detections[overlapping, 16:17]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(dim=0) / total_score
                weighted_detection[:16] = weighted
                weighted_detection[16] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections    


# IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py

def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes): return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)

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

gpu = "cpu"

back_net = BlazeFace().to(gpu)

back_net.load_state_dict(torch.load("blazefaceback.pth"))
back_net.eval() 

back_net.anchors = torch.tensor(np.load("anchorsback.npy"), dtype=torch.float32, device=back_net._device())

back_net.min_score_thresh = 0.75
back_net.min_suppression_threshold = 0.3

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

detections = back_net.predict_on_image(img).numpy()

expected = [[0.22293027,0.3687327,0.35492355,0.500726,0.4048541,0.253551,0.45936358,0.25396332,0.42835188,0.2809909,0.42859644,0.31245646,0.37655264,0.27385083,0.49636966,0.27672035,0.83855903,],
[0.30805102,0.68929595,0.42866126,0.8099063,0.71050656,0.34094658,0.75901216,0.34136337,0.7211923,0.3699867,0.7258061,0.3949228,0.703986,0.3506133,0.8086657,0.3542543,0.7997207,],]

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