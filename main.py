# Import necessary libraries
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
from yolov6.utils.events import LOGGER
from yolov6.core.inferer_starfish import Inferer

# PyQt5 imports for creating the GUI
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from concurrent.futures import ThreadPoolExecutor

# Set the root directory and add it to sys.path if not already included
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def preprocess_image(img):
    """
    Preprocess the image for input into the model.
    
    Args:
        img (numpy.ndarray): The input image.
    
    Returns:
        torch.Tensor: The preprocessed image as a tensor.
    """
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and transpose dimensions
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0  # Normalize the image
    return torch.from_numpy(img).unsqueeze(0)  # Convert to tensor and add a batch dimension

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (tuple): The first bounding box in (x1, y1, x2, y2) format.
        box2 (tuple): The second bounding box in (x1, y1, x2, y2) format.
    
    Returns:
        float: The IoU between the two bounding boxes.
    """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Compute the area of intersection
    inter_area = max(0, min(x2, x2_p) - max(x1, x1_p)) * max(0, min(y2, y2_p) - max(y1, y1_p))
    
    if inter_area == 0:
        return 0.0
    
    # Compute the area of each box
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    
    # Compute the union area and IoU
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def merge_boxes(faster_rcnn_boxes, yolov6_boxes, iou_threshold=0.5):
    """
    Merge bounding boxes from Faster R-CNN and YOLOv6 using IoU.
    
    Args:
        faster_rcnn_boxes (list): List of bounding boxes from Faster R-CNN.
        yolov6_boxes (list): List of bounding boxes from YOLOv6.
        iou_threshold (float): The IoU threshold for merging boxes.
    
    Returns:
        numpy.ndarray: The merged bounding boxes.
    """
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

    # Add unique boxes that were not merged
    merged_boxes.extend(list(unique_faster_rcnn_boxes))
    merged_boxes.extend(list(unique_yolov6_boxes))
    return np.array(merged_boxes)

class Ui_MainWindow(object):
    """
    Class to define the main window UI for the PyQt5 application.
    """
    def setupUi(self, MainWindow):
        """
        Setup the main window layout and widgets.
        
        Args:
            MainWindow (QMainWindow): The main window object.
        """
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(960, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Setup buttons, labels, and text edits
        self.run_button = QtWidgets.QPushButton(self.centralwidget)
        self.run_button.setGeometry(QtCore.QRect(180, 90, 600, 40))
        self.run_button.setObjectName("run_button")
        self.run_button.clicked.connect(self.run_program)
        
        self.Label = QtWidgets.QLabel(self.centralwidget)
        self.Label.setGeometry(QtCore.QRect(0, 20, 960, 40))
        self.Label.setObjectName("Label")
        
        # Define text edit fields and labels for various parameters
        self.yolockpt = QtWidgets.QTextEdit(self.centralwidget)
        self.yolockpt.setGeometry(QtCore.QRect(330, 210, 560, 40))
        self.yolockpt.setObjectName("yolockpt")
        
        self.yolo_conf = QtWidgets.QTextEdit(self.centralwidget)
        self.yolo_conf.setGeometry(QtCore.QRect(330, 260, 560, 40))
        self.yolo_conf.setObjectName("yolo_conf")
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 210, 250, 40))
        self.label.setObjectName("label")
        
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 260, 250, 40))
        self.label_2.setObjectName("label_2")
        
        self.Device = QtWidgets.QTextEdit(self.centralwidget)
        self.Device.setGeometry(QtCore.QRect(330, 560, 560, 40))
        self.Device.setObjectName("Device")
        
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(40, 560, 250, 40))
        self.label_4.setObjectName("label_4")
        
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(40, 610, 250, 40))
        self.label_5.setObjectName("label_5")
        
        self.overlap_threshold = QtWidgets.QTextEdit(self.centralwidget)
        self.overlap_threshold.setGeometry(QtCore.QRect(330, 610, 560, 40))
        self.overlap_threshold.setObjectName("overlap_threshold")
        
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(40, 310, 250, 40))
        self.label_6.setObjectName("label_6")
        
        self.fasterrcnn_ckpt_file = QtWidgets.QTextEdit(self.centralwidget)
        self.fasterrcnn_ckpt_file.setGeometry(QtCore.QRect(330, 310, 560, 40))
        self.fasterrcnn_ckpt_file.setObjectName("fasterrcnn_ckpt_file")
        
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(40, 360, 250, 40))
        self.label_7.setObjectName("label_7")
        
        self.fasterrcnn_conf = QtWidgets.QTextEdit(self.centralwidget)
        self.fasterrcnn_conf.setGeometry(QtCore.QRect(330, 360, 560, 40))
        self.fasterrcnn_conf.setObjectName("fasterrcnn_conf")
        
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(40, 460, 250, 40))
        self.label_8.setObjectName("label_8")
        
        self.output_vid = QtWidgets.QTextEdit(self.centralwidget)
        self.output_vid.setGeometry(QtCore.QRect(330, 460, 560, 40))
        self.output_vid.setObjectName("output_vid")
        
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(40, 410, 250, 40))
        self.label_9.setObjectName("label_9")
        
        self.input_vid = QtWidgets.QTextEdit(self.centralwidget)
        self.input_vid.setGeometry(QtCore.QRect(330, 410, 560, 40))
        self.input_vid.setObjectName("input_vid")
        
        self.output_csv = QtWidgets.QTextEdit(self.centralwidget)
        self.output_csv.setGeometry(QtCore.QRect(330, 510, 560, 40))
        self.output_csv.setObjectName("output_csv")
        
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(40, 510, 250, 40))
        self.label_10.setObjectName("label_10")
        
        self.frame_rate = QtWidgets.QTextEdit(self.centralwidget)
        self.frame_rate.setGeometry(QtCore.QRect(330, 660, 560, 40))
        self.frame_rate.setObjectName("frame_rate")
        
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(40, 660, 250, 40))
        self.label_11.setObjectName("label_11")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 960, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # Retranslate the UI to set text for widgets
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def run_program(self):
        """
        Method to run the main program when the button is clicked.
        """
        MainWindow.close()
        
        main(
            osp.join(self.yolockpt.toPlainText()), 
            self.Device.toPlainText(), 
            float(self.yolo_conf.toPlainText()), 
            self.input_vid.toPlainText(), 
            self.output_vid.toPlainText(), 
            float(self.overlap_threshold.toPlainText()), 
            self.output_csv.toPlainText(), 
            float(self.fasterrcnn_conf.toPlainText()), 
            int(self.frame_rate.toPlainText()), 
            osp.join(self.fasterrcnn_ckpt_file.toPlainText())
        )
        
    def retranslateUi(self, MainWindow):
        """
        Set the text for UI elements.
        
        Args:
            MainWindow (QMainWindow): The main window object.
        """
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.run_button.setText(_translate("MainWindow", "Run Program"))
        self.Label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt;\">Set Parameters and Click Run Program to Begin</span></p></body></html>"))
        self.yolockpt.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\"><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">C:\\\\Users\\\\ianki\\\\Documents\\\\2023-2024_coding\\\\AI_studies\\\\starfish_final\\\\last_ckpt.pt</p></body></html>"))
        self.yolo_conf.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\"><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">0.33</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">YOLO Ckpt File Location</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">YOLO Conf Threshold: </span></p></body></html>"))
        self.Device.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\"><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">0</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Device</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">IOU Threshold</span></p></body></html>"))
        self.overlap_threshold.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\"><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">0.4</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Faster Rcnn Ckpt File Location</span></p></body></html>"))
        self.fasterrcnn_ckpt_file.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\"><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">C:\\\\Users\\\\ianki\\\\Documents\\\\2023-2024_coding\\\\AI_studies\\\\starfish_final\\\\best_brittle_star_model6.pth</p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Faster Rcnn Conf Threshold</span></p></body></html>"))
        self.fasterrcnn_conf.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\"><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">0.31</span></p></body></html>"))
        self.label_8.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Output Video File Location</span></p></body></html>"))
        self.output_vid.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\"><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">C:\\\\Users\\\\ianki\\\\Documents\\\\2023-2024_coding\\\\AI_studies\\\\starfish_final\\\\starfish_finder.avi</p></body></html>"))
        self.label_9.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Input Video File Location</span></p></body></html>"))
        self.input_vid.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\"><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">C:\\\\Users\\\\ianki\\\\Documents\\\\2023-2024_coding\\\\AI_studies\\\\starfish_final\\\\seafloor_footage.avi</p></body></html>"))
        self.output_csv.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\"><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">C:\\\\Users\\\\ianki\\\\Documents\\\\2023-2024_coding\\\\AI_studies\\\\starfish_final\\\\starfish_bboxes.csv</p></body></html>"))
        self.label_10.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Output BBox Csv Location</span></p></body></html>"))
        self.frame_rate.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\"><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">30</span></p></body></html>"))
        self.label_11.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Tracker Update Frame Rate</span></p></body></html>"))

def main(weights, device, conf_threshold, video_file, output_file, iou_threshold, csv_file_path, rcnn_threshold, frame_rate, faster_rcnn_location):
    """
    Main function to run the detection and tracking process.
    
    Args:
        weights (str): Path to the YOLOv6 weights file.
        device (str): Device to run the models on (e.g., 'cpu' or GPU ID).
        conf_threshold (float): Confidence threshold for YOLOv6.
        video_file (str): Path to the input video file.
        output_file (str): Path to save the output video file.
        iou_threshold (float): IoU threshold for merging boxes.
        csv_file_path (str): Path to save the output CSV file with bounding boxes.
        rcnn_threshold (float): Confidence threshold for Faster R-CNN.
        frame_rate (int): Frame rate to update the tracker.
        faster_rcnn_location (str): Path to the Faster R-CNN checkpoint file.
    """
    img_size = 640
    iou_thres = 0.45
    max_det = 1000
    agnostic_nms = False
    half = False
    overlap_threshold = 50

    # Initialize the Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load(faster_rcnn_location))
    if device != 'cpu':
        device = int(device)
    model.to(device)
    model.eval()
    
    # Initialize the YOLOv6 Inferer
    inferer = Inferer(weights, device, img_size, half)

    # Open the input video file
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    ind = 0
    timer = frame_rate + 1
    saved_bboxes = []  # List to store bounding box coordinates

    size = (frame_width, frame_height)

    # Initialize the video writer
    result = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), 30, size)

    # Define OpenCV object trackers
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.legacy.TrackerCSRT_create,
        "kcf": cv2.legacy.TrackerKCF_create,
        "mil": cv2.legacy.TrackerMIL_create,
    }

    cv2.namedWindow('output')
    trackers = cv2.legacy.MultiTracker_create()

    while cap.isOpened():
        ret, frame = cap.read()
        np_frame = np.asarray(copy.deepcopy(frame))  # Image with bounding box on it
        ai_frame = np.asarray(copy.deepcopy(frame))
        try:
            H, W = np_frame.shape[0:2]
        except:
            break
        cv2.resizeWindow('output', W, H)

        if timer <= frame_rate:
            # Update the trackers and draw bounding boxes on the frame
            success, boxes = trackers.update(frame)

            if success:
                px, py, pw, ph = [], [], [], []
                for i, box in enumerate(boxes):
                    x, y, w, h = [int(v) for v in box]
                    if 0 <= x < W and 0 <= y < H and x + w <= W and y + h <= H:
                        px.append(x)
                        py.append(y)
                        pw.append(w)
                        ph.append(h)
                        cv2.rectangle(np_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        saved_bboxes.append([ind + 1, x, x + w, y, y + h])

            if not success:
                print('failure')
                trackers = cv2.legacy.MultiTracker_create()
                for index in range(len(px)):
                    if px[index] + pw[index] < W and px[index] > 0 and py[index] + ph[index] < H and py[index] > 0:
                        roi2 = (px[index], py[index], pw[index], ph[index])
                        tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                        trackers.add(tracker, frame, roi2)

                bbox_trackers = trackers.getObjects()
                for i, box in enumerate(bbox_trackers):
                    x, y, w, h = [int(v) for v in box]
                    cv2.rectangle(np_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    saved_bboxes.append([ind + 1, x, x + w, y, y + h])

        else:
            # Update the trackers and draw bounding boxes on the frame
            success, boxes = trackers.update(frame)

            if success:
                px, py, pw, ph = [], [], [], []
                for i, box in enumerate(boxes):
                    x, y, w, h = [int(v) for v in box]
                    if 0 <= x < W and 0 <= y < H and x + w <= W and y + h <= H:
                        px.append(x)
                        py.append(y)
                        pw.append(w)
                        ph.append(h)
                        cv2.rectangle(np_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        saved_bboxes.append([ind + 1, x, x + w, y, y + h])

            if not success:
                print('tracking reset')
                trackers = cv2.legacy.MultiTracker_create()
                for index in range(len(px)):
                    if px[index] + pw[index] < W and px[index] > 0 and py[index] + ph[index] < H and py[index] > 0:
                        roi2 = (px[index], py[index], pw[index], ph[index])
                        tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                        trackers.add(tracker, frame, roi2)

                bbox_trackers = trackers.getObjects()
                for i, box in enumerate(bbox_trackers):
                    x, y, w, h = [int(v) for v in box]
                    cv2.rectangle(np_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    saved_bboxes.append([ind + 1, x, x + w, y, y + h])

            # Run YOLOv6 and Faster R-CNN detection in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_yolo = executor.submit(inferer.inferv2, ai_frame, 0.2, iou_thres, None, agnostic_nms, max_det, conf_threshold, overlap_threshold)
                future_rcnn = executor.submit(model, preprocess_image(ai_frame).to(device))
                yolo_boxes = future_yolo.result()
                faster_rcnn_outputs = future_rcnn.result()

            yolo_boxes = [box[1] for box in yolo_boxes if box[0] == 1]

            # Get the bounding boxes from Faster R-CNN outputs
            faster_rcnn_boxes = faster_rcnn_outputs[0]['boxes'].cpu().detach().numpy()
            faster_rcnn_scores = faster_rcnn_outputs[0]['scores'].cpu().detach().numpy()
            faster_rcnn_boxes = faster_rcnn_boxes[faster_rcnn_scores > rcnn_threshold].astype(int)

            # Merge YOLOv6 and Faster R-CNN boxes
            merged_boxes = merge_boxes(faster_rcnn_boxes, yolo_boxes, iou_threshold)

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

        # Display the output frame
        cv2.imshow('output', np_frame)
        cv2.waitKey(30)
        
        # Write the frame to the output video file
        result.write(np_frame)
        
        print(ind)
        timer += 1
        ind += 1

    # Save bounding box coordinates to CSV file
    my_df = pd.DataFrame(saved_bboxes, columns=['Current Frame', 'X Bound, Left', 'X Bound, Right', 'Y Bound, Upper', 'Y Bound, Lower'])
    my_df.to_csv(csv_file_path, index=False)

    # Release resources
    cap.release()
    result.release()
    cv2.destroyAllWindows()

# Create the application and the main window
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()

# Setup the UI and show the main window
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()

# Start the application event loop
sys.exit(app.exec_())
