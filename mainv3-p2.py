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


    #FAST_RCNN INIT

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
    result = cv2.VideoWriter(output_file,  cv2.VideoWriter_fourcc(*'MJPG'), 30, size)

    OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
	"kcf": cv2.legacy.TrackerKCF_create,
	"mil": cv2.legacy.TrackerMIL_create,
    }

    cv2.namedWindow('output')
    trackers = cv2.legacy.MultiTracker_create()
    

    
    while cap.isOpened():
        ret, frame = cap.read()

        np_frame = np.asarray(copy.deepcopy(frame)) # image with bounding box on it
        ai_frame = np.asarray(copy.deepcopy(frame))

        H, W = np_frame.shape[0:2]
        cv2.resizeWindow('output', W, H)


        if timer <= frame_rate:


            (success, boxes) = trackers.update(frame)

            if success == True:
                px = []
                py = []
                pw = []
                ph = []

                for i,box in enumerate(boxes):

                    
                    (x, y, w, h) = [int(v) for v in box]
                    if x+w > W or x < 0 or y+h > H+0 or y < 0:
                        continue
                    else:

                        px.append(x)
                        py.append(y)
                        pw.append(w)
                        ph.append(h)
                        cv2.rectangle(np_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        saved_bboxes.append([ind+1, x, x+w, y, y+h])

            if success == False:
                print('failure')


                trackers = cv2.legacy.MultiTracker_create()
                for index in range(0, len(px)):
                    if px[index]+pw[index] < W and px[index] > 0 and py[index]+ph[index] < H+0 and py[index] > 0:
                        roi2 = (px[index], py[index], pw[index], ph[index])
                        tracker = tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                        trackers.add(tracker, frame, roi2)

                bbox_trackers = trackers.getObjects()
                for i,box in enumerate(bbox_trackers):
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(np_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    saved_bboxes.append([ind+1, x, x+w, y, y+h])


                

                
    
        else:
            print('stuff')
            overlap_checker = np.zeros([np_frame.shape[0], np_frame.shape[1]], dtype=np.uint8) # for yolo and faster_rcnn

            (success, boxes) = trackers.update(frame)

            if success == True:
                px = []
                py = []
                pw = []
                ph = []
                for i,box in enumerate(boxes):
                    (x, y, w, h) = [int(v) for v in box]
                    if x+w > W or x < 0 or y+h > H+0 or y < 0:
                        continue
                    else:
                        px.append(x)
                        py.append(y)
                        pw.append(w)
                        ph.append(h)
                        overlap_checker[y:y+h, x:x+h] = 255
                        cv2.rectangle(np_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        saved_bboxes.append([ind+1, x, x+w, y, y+h])


            if success == False:
                print('failure')
                



                trackers = cv2.legacy.MultiTracker_create()
                for index in range(0, len(px)):
                    if px[index]+pw[index] < W and px[index] > 0 and py[index]+ph[index] < H+0 and py[index] > 0:
                        
                        roi2 = (px[index], py[index], pw[index], ph[index])
                        tracker = tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                        trackers.add(tracker, frame, roi2)

                bbox_trackers = trackers.getObjects()
                for i,box in enumerate(bbox_trackers):
                    (x, y, w, h) = [int(v) for v in box]
                    overlap_checker[y:y+h, x:x+h] = 255
                    cv2.rectangle(np_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    saved_bboxes.append([ind+1, x, x+w, y, y+h])


            box_checker = copy.deepcopy(overlap_checker)
            yolo_boxes = []

            boxes = inferer.inferv2(ai_frame, conf_thres, iou_thres, None, agnostic_nms, max_det, conf_threshold, overlap_threshold)
            for line in boxes:

                x1, y1, x2, y2 = line[1]


                block = box_checker[y1:y2, x1:x2].astype(np.float32)
                block_mean = np.mean(block)
                
                if block_mean > overlap_threshold:
                    continue
                box_checker[y1:y2, x1:x2] = 255

                

                roi = (x1, y1, x2-x1, y2-y1)
                yolo_boxes.append(roi)

            

            
            image = preprocess_image(ai_frame).to(device)
            with torch.no_grad():
                outputs = model(image)
            boxes = outputs[0]['boxes'].cpu().numpy()
            scores = outputs[0]['scores'].cpu().numpy()
            labels = outputs[0]['labels'].cpu().numpy()


            for box, score, label in zip(boxes, scores, labels):
                if score > rcnn_threshold:
                    x1, y1, x2, y2 = box.astype(int)
                    

                    block = box_checker[y1:y2, x1:x2].astype(np.float32)
                    block_mean = np.mean(block)
                    
                    if block_mean > overlap_threshold:
                        continue
                    box_checker[y1:y2, x1:x2] = 255
                    overlap_checker[y1:y2, x1:x2] = 255
                    tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                    roi = (x1, y1, x2-x1, y2-y1)
                    trackers.add(tracker, frame, roi)
                    if ind == 0:
                        cv2.rectangle(np_frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                        saved_bboxes.append([ind+1, x1, x2, y1, y2])
            

            for n in yolo_boxes:
                x, y, w, h = n
                block = overlap_checker[y:y+h, x:x+h].astype(np.float32)
                block_mean = np.mean(block)
                
                if block_mean > overlap_threshold:
                    continue

                overlap_checker[y:y+h, x:x+h] = 255
                tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                trackers.add(tracker, frame, n)
                if ind == 0:
                    cv2.rectangle(np_frame, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                    saved_bboxes.append([ind+1, x, y, x+w, y+h])

                    
            timer = 0





        cv2.imshow('output', np_frame)
        cv2.waitKey(30)
        
        result.write(np_frame)
        
        print(ind)
        timer += 1
        ind += 1

    
    my_df = pd.DataFrame(saved_bboxes, columns=['Current Frame','X Bound, Left','X Bound, Right','Y Bound, Upper', 'Y Bound, Lower'])
    my_df.to_csv(csv_file_path, index=False)  # saving bbox coords to csv file

    cap.release()
    result.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    main()
