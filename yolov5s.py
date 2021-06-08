
import torch
import torchvision
from torch import nn
import cv2
import numpy as np
import time
from collections import OrderedDict

#---network definition
def autopad(k, p=None):  # kernel, padding
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.act(self.bn(self.conv(x)))
        return y

class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        y = self.conv(torch.cat([x[:, :, ::2, ::2], x[:, :, 1::2, ::2], x[:, :, ::2, 1::2], x[:, :, 1::2, 1::2]], 1))
        return y

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        return y

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        return y

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        y = self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
        return y

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        y = torch.cat(x, self.d)
        return y

class Detect(nn.Module):
    stride = torch.tensor(data=[8., 16., 32.])  # strides computed during build

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # inference output

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if True:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]: self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                y[:, :, :, :, 0:2] = (y[:, :, :, :, 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy offset ralative to grid
                y[:, :, :, :, 2:4] = (y[:, :, :, :, 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return torch.cat(z, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

#---utility function
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=()):
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def plot_one_box(x, im, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

class YoloV5(nn.Module):

    def __init__(self, nc=80, anchors=None):
        super(YoloV5, self).__init__()
        assert anchors != None, 'anchor must be provided'

        self.Focus_0 = Focus(c1=3, c2=32, k=3, p=1)
        self.Conv_1 = Conv(c1=32, c2=64, k=3, s=2, p=1)
        self.C3_2 = C3(c1=64, c2=64)
        self.Conv_3 = Conv(c1=64, c2=128, k=3, s=2, p=1)
        self.C3_4 = C3(c1=128, c2=128, n=3)
        self.Conv_5 = Conv(c1=128, c2=256, k=3, s=2, p=1)
        self.C3_6 = C3(c1=256, c2=256, n=3)
        self.Conv_7 = Conv(c1=256, c2=512, k=3, s=2, p=1)
        self.SPP_8 = SPP(c1=512, c2=512, k=(5, 9, 13))
        self.C3_9 = C3(c1=512, c2=512)
        self.Conv_10 = Conv(c1=512, c2=256, k=1, s=1)
        self.Upsample_11 = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.Concat_12 = Concat(dimension=1)
        self.C3_13 = C3(c1=512, c2=256, shortcut=False)
        self.Conv_14 = Conv(c1=256, c2=128, k=1, s=1)
        self.Upsample_15 = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.Concat_16 = Concat(dimension=1)
        self.C3_17 = C3(c1=256, c2=128, shortcut=False)
        self.Conv_18 = Conv(c1=128, c2=128, k=3, s=2, p=1)
        self.Concat_19 = Concat(dimension=1)
        self.C3_20 = C3(c1=256, c2=256, shortcut=False)
        self.Conv_21 = Conv(c1=256, c2=256, k=3, s=2, p=1)
        self.Concat_22 = Concat(dimension=1)
        self.C3_23 = C3(c1=512, c2=512, shortcut=False)
        self.Detect_24 = Detect(nc=nc, anchors=anchors, ch=(128, 256, 512))

    def forward(self, x):
        # x_0 = self.Focus_0(x)
        x_0 = self.Focus_0.conv(torch.cat([x[:, :, ::2, ::2], x[:, :, 1::2, ::2], x[:, :, ::2, 1::2], x[:, :, 1::2, 1::2]], 1))
        x_1 = self.Conv_1.act(self.Conv_1.bn(self.Conv_1.conv(x_0)))
        x_2 = self.C3_2(x_1)
        x_3 = self.Conv_3(x_2)
        x_4 = self.C3_4(x_3)
        x_5 = self.Conv_5(x_4)
        x_6 = self.C3_6(x_5)
        x_7 = self.Conv_7(x_6)
        x_8 = self.SPP_8(x_7)
        x_9 = self.C3_9(x_8)
        x_10 = self.Conv_10(x_9)
        x_11 = self.Upsample_11(x_10)
        x_12 = self.Concat_12([x_11, x_6])
        x_13 = self.C3_13(x_12)
        x_14 = self.Conv_14(x_13)
        x_15 = self.Upsample_15(x_14)
        x_16 = self.Concat_16([x_15, x_4])
        x_17 = self.C3_17(x_16)
        x_18 = self.Conv_18(x_17)
        x_19 = self.Concat_19([x_18, x_14])
        x_20 = self.C3_20(x_19)
        x_21 = self.Conv_21(x_20)
        x_22 = self.Concat_22([x_21, x_10])
        x_23 = self.C3_23(x_22)
        y = self.Detect_24([x_17, x_20, x_23])
        return y

#------------------------main
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
         'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
         'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
         'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
         'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
         'wine glass', 'cup', 'fork', 'knife', 'spoon',
         'bowl', 'banana', 'apple', 'sandwich',
         'orange', 'broccoli', 'carrot', 'hot dog',
         'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
         'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
#
anchors = [[10, 13, 16, 30, 33, 23],
           [30, 61, 62, 45, 59, 119],
           [116, 90, 156, 198, 373, 326]]
conf_thres = 0.25
iou_thres = 0.45
classes = None
agnostic_nms = False
hide_labels = False
hide_conf = False
line_thickness = 2
#
model = YoloV5(anchors=anchors)
# load parameter
model_pre_dict = torch.load('yolov5s.dict')
# convert key name
model_pre_key_list = list(model_pre_dict.keys())
#
model_parameter_dict = OrderedDict()
# copy parameter
for key_index, key in enumerate(model.state_dict().keys()): model_parameter_dict[key] = model_pre_dict[model_pre_key_list[key_index]]
model.load_state_dict(model_parameter_dict)
model.eval()
print('model parameter loaded')
#
image = cv2.imread('demo.jpg')
image = cv2.resize(image,(640,480))
image_init = image.copy()
image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
image = np.ascontiguousarray(image)
image = torch.from_numpy(image)
image = image.float()  # uint8 to fp16/32
image /= 255.0  # 0 - 255 to 0.0 - 1.0
if image.ndimension() == 3: image = image.unsqueeze(0)
print('--------------------********************--------------------')
print('image.shape', image.shape)
with torch.no_grad(): pred = model(image)
print('pred.shape', pred.shape)
print('--------------------********************--------------------')
# Apply NMS
pred = non_max_suppression(prediction=pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes, agnostic=agnostic_nms)
print('pred[0].shape', pred[0].shape)
det = pred[0]
print('det.shape', det.shape)
print('det = ', det)
if len(det):
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(image.shape[2:], det[:, :4], image_init.shape).round()
    # Write results
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)  # integer class
        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
        plot_one_box(xyxy, image_init, label=label, color=colors[c], line_thickness=line_thickness)

cv2.imshow('image_init', image_init)
cv2.waitKey()