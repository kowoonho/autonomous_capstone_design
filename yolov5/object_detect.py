import os
import sys
from pathlib import Path


import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import scale_coords, non_max_suppression

from utils.torch_utils import select_device, smart_inference_mode

labels_to_names = {0:'Stop_Sign'}


@smart_inference_mode()
def run(
        image, #image array type
        weights=ROOT/ 'yolov5s.pt',  # model.pt path(s)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
):

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)

    # image preprocessing
    draw_img = image.copy()
    draw_img = cv2.resize(draw_img, imgsz)

    image = torch.from_numpy(image).to(device)
    image = image.half() if model.fp16 else image.float()  # uint8 to fp16/32
    image /= 255
    image = image.permute(2, 0, 1)
    transform = transforms.Resize(imgsz)
    image = transform(image)

    if len(image.shape) == 3:
        image = image[None]
    pred = model(image, augment=augment)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    pred = pred[0]
    green_color = (0, 255, 0)
    red_color = (255, 0, 0)
    print(pred)

    # image drawing
    for *box, cf, cls in pred:
        cf = cf.item()
        cls = cls.item()

        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

        ##################################
        area_threshold = 1 # 설정해야할 값
        bbox_area = (p2[0] - p1[0]) * (p2[1] - p1[1])
        if bbox_area > area_threshold:
            print("{}'s bbox area : {}".format(labels_to_names[cls], bbox_area))

        

        ##################################


        caption = "{}: {:.4f}".format(labels_to_names[cls], cf)
        cv2.rectangle(draw_img, p1, p2, color = green_color, thickness = 2)
        cv2.putText(draw_img, caption, (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, thickness = 1)
        print(caption)
    
    plt.figure(figsize=(12,12))
    plt.imshow(draw_img)


