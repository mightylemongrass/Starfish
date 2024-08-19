import os
import sys
import os.path as osp
import cv2
import numpy as np
import yaml
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import pandas as pd
import copy

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.utils.events import LOGGER
from yolov6.core.inferer_starfish import Inferer

def preprocess_image(img):
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)

# Function to compute IoU
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_area = max(0, min(x2, x2_p) - max(x1, x1_p)) * max(0, min(y2, y2_p) - max(y1, y1_p))
    if inter_area == 0:
        return 0.0
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

# Function to merge boxes
def merge_boxes(faster_rcnn_boxes, yolov6_boxes, iou_threshold=0.5):
    merged_boxes = []
    unique_faster_rcnn_boxes = set(tuple(box) for box in faster_rcnn_boxes)
    unique_yolov6_boxes = set(tuple(box) for box in yolov6_boxes)

    for fr_box in faster_rcnn_boxes:
        for yolo_box in yolov6_boxes:
            if compute_iou(fr_box, yolo_box) > iou_threshold:
                merged_boxes.append([
                    min(fr_box[0], yolo_box[0]),
                    min(fr_box[1], yolo_box[1]),
                    max(fr_box[2], yolo_box[2]),
                    max(fr_box[3], yolo_box[3])
                ])
                unique_faster_rcnn_boxes.discard(tuple(fr_box))
                unique_yolov6_boxes.discard(tuple(yolo_box))

    merged_boxes.extend(list(unique_faster_rcnn_boxes))
    merged_boxes.extend(list(unique_yolov6_boxes))
    return np.array(merged_boxes)

def main():
    with open('parameters.yaml', 'r') as file:
        data = yaml.safe_load(file)

    weights = osp.join(data['YOLO_file_location'])
    img_size = data['img_size']
    conf_thres = data['conf_thres']
    iou_thres = data['iou_thres']
    max_det = data['max_det']
    device = data['device']
    agnostic_nms = data['agnostic_nms']
    half = data['half']
    conf_threshold = data['yolo_conf_threshold']
    video_file = data['video_file']
    output_file = data['output_file_location']
    overlap_threshold = data['overlap_threshold']
    csv_file_path = data['csv_filepath']
    rcnn_threshold = data['faster_rcnn_conf_threshold']
    frame_rate = data['frame_rate_of_program']
    faster_rcnn_location = data['faster_rcnn_location']

    # FAST_RCNN INIT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load(faster_rcnn_location))
    model.to(device)
    model.eval()
    
    inferer = Inferer(weights, device, img_size, half)

    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    ind = 0
    timer = frame_rate+1
    saved_bboxes = [] # saved bboxes

    size = (frame_width, frame_height)
    result = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), 30, size)

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.legacy.TrackerCSRT_create,
        "kcf": cv2.legacy.TrackerKCF_create,
        "mil": cv2.legacy.TrackerMIL_create,
    }

    cv2.namedWindow('output')
    trackers = cv2.legacy.MultiTracker_create()

    while cap.isOpened():
        ret, frame = cap.read()
        np_frame = np.asarray(copy.deepcopy(frame))  # image with bounding box on it
        ai_frame = np.asarray(copy.deepcopy(frame))
        H, W = np_frame.shape[0:2]
        cv2.resizeWindow('output', W, H)

        if timer <= frame_rate:
            (success, boxes) = trackers.update(frame)

            if success:
                for i, box in enumerate(boxes):
                    (x, y, w, h) = [int(v) for v in box]
                    if 0 <= x < W and 0 <= y < H and x + w <= W and y + h <= H:
                        cv2.rectangle(np_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        saved_bboxes.append([ind + 1, x, x + w, y, y + h])

            if not success:
                trackers = cv2.legacy.MultiTracker_create()
                for box in boxes:
                    x, y, w, h = [int(v) for v in box]
                    if 0 <= x < W and 0 <= y < H and x + w <= W and y + h <= H:
                        tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                        trackers.add(tracker, frame, (x, y, w, h))
                        cv2.rectangle(np_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        saved_bboxes.append([ind + 1, x, x + w, y, y + h])

        else:
            overlap_checker = np.zeros([np_frame.shape[0], np_frame.shape[1]], dtype=np.uint8)

            (success, boxes) = trackers.update(frame)

            if success:
                for i, box in enumerate(boxes):
                    (x, y, w, h) = [int(v) for v in box]
                    if 0 <= x < W and 0 <= y < H and x + w <= W and y + h <= H:
                        overlap_checker[y:y+h, x:x+w] = 255
                        cv2.rectangle(np_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        saved_bboxes.append([ind + 1, x, x + w, y, y + h])

            if not success:
                trackers = cv2.legacy.MultiTracker_create()
                for box in boxes:
                    x, y, w, h = [int(v) for v in box]
                    if 0 <= x < W and 0 <= y < H and x + w <= W and y + h <= H:
                        tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                        trackers.add(tracker, frame, (x, y, w, h))
                        cv2.rectangle(np_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        saved_bboxes.append([ind + 1, x, x + w, y, y + h])

            # YOLOv6 detection
            yolo_boxes = []
            boxes = inferer.inferv2(ai_frame, conf_thres, iou_thres, None, agnostic_nms, max_det, conf_threshold, overlap_threshold)
            for line in boxes:
                x1, y1, x2, y2 = line[1]

                block = overlap_checker[y1:y2, x1:x2].astype(np.float32)
                block_mean = np.mean(block)
                
                if block_mean <= overlap_threshold:
                    yolo_boxes.append((x1, y1, x2, y2))

            # Faster R-CNN detection
            image = preprocess_image(ai_frame).to(device)
            with torch.no_grad():
                outputs = model(image)
            faster_rcnn_boxes = outputs[0]['boxes'].cpu().numpy()
            faster_rcnn_scores = outputs[0]['scores'].cpu().numpy()

            faster_rcnn_boxes = faster_rcnn_boxes[faster_rcnn_scores > rcnn_threshold]
            faster_rcnn_boxes = [box.astype(int) for box in faster_rcnn_boxes]

            # Merge YOLOv6 and Faster R-CNN boxes
            merged_boxes = merge_boxes(faster_rcnn_boxes, yolo_boxes)

            # Initialize trackers with merged boxes
            trackers = cv2.legacy.MultiTracker_create()
            for box in merged_boxes:
                tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                x1, y1, x2, y2 = box.astype(int)
                roi = (x1, y1, x2 - x1, y2 - y1)
                trackers.add(tracker, frame, roi)
                cv2.rectangle(np_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                saved_bboxes.append([ind + 1, x1, x2, y1, y2])

            timer = 0

        cv2.imshow('output', np_frame)
        cv2.waitKey(30)
        
        result.write(np_frame)
        
        print(ind)
        timer += 1
        ind += 1

    my_df = pd.DataFrame(saved_bboxes, columns=['Current Frame', 'X Bound, Left', 'X Bound, Right', 'Y Bound, Upper', 'Y Bound, Lower'])
    my_df.to_csv(csv_file_path, index=False)  # saving bbox coords to csv file

    cap.release()
    result.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
