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
import pandas as pd
import copy
from yolov6.core.inferer_starfish import Inferer
import argparse

# PyQt5 imports for creating the GUI
from PyQt5 import QtCore, QtWidgets
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
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        self.Label = QtWidgets.QLabel(self.centralwidget)
        self.Label.setObjectName("Label")
        self.gridLayout.addWidget(self.Label, 0, 0, 1, 2)

        self.fasterrcnn_conf = QtWidgets.QTextEdit(self.centralwidget)
        self.fasterrcnn_conf.setObjectName("fasterrcnn_conf")
        self.gridLayout.addWidget(self.fasterrcnn_conf, 5, 1, 1, 1)

        self.input_vid = QtWidgets.QTextEdit(self.centralwidget)
        self.input_vid.setObjectName("input_vid")
        self.gridLayout.addWidget(self.input_vid, 6, 1, 1, 1)

        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 5, 0, 1, 1)

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)

        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 12, 0, 1, 1)

        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 11, 0, 1, 1)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)

        self.fasterrcnn_ckpt_file = QtWidgets.QTextEdit(self.centralwidget)
        self.fasterrcnn_ckpt_file.setObjectName("fasterrcnn_ckpt_file")
        self.gridLayout.addWidget(self.fasterrcnn_ckpt_file, 4, 1, 1, 1)

        self.Device = QtWidgets.QTextEdit(self.centralwidget)
        self.Device.setObjectName("Device")
        self.gridLayout.addWidget(self.Device, 9, 1, 1, 1)

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)

        self.output_vid = QtWidgets.QTextEdit(self.centralwidget)
        self.output_vid.setObjectName("output_vid")
        self.gridLayout.addWidget(self.output_vid, 7, 1, 1, 1)

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 10, 0, 1, 1)

        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 6, 0, 1, 1)

        self.frame_rate = QtWidgets.QTextEdit(self.centralwidget)
        self.frame_rate.setObjectName("frame_rate")
        self.gridLayout.addWidget(self.frame_rate, 11, 1, 1, 1)

        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 8, 0, 1, 1)

        self.yolockpt = QtWidgets.QTextEdit(self.centralwidget)
        self.yolockpt.setObjectName("yolockpt")
        self.gridLayout.addWidget(self.yolockpt, 2, 1, 1, 1)

        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 7, 0, 1, 1)

        self.yolo_conf = QtWidgets.QTextEdit(self.centralwidget)
        self.yolo_conf.setObjectName("yolo_conf")
        self.gridLayout.addWidget(self.yolo_conf, 3, 1, 1, 1)

        self.overlap_threshold = QtWidgets.QTextEdit(self.centralwidget)
        self.overlap_threshold.setObjectName("overlap_threshold")
        self.gridLayout.addWidget(self.overlap_threshold, 10, 1, 1, 1)

        self.output_xls = QtWidgets.QTextEdit(self.centralwidget)
        self.output_xls.setObjectName("output_xls")
        self.gridLayout.addWidget(self.output_xls, 8, 1, 1, 1)

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 9, 0, 1, 1)

        self.run_button = QtWidgets.QPushButton(self.centralwidget)
        self.run_button.setObjectName("run_button")
        self.gridLayout.addWidget(self.run_button, 1, 0, 1, 2)
        self.run_button.clicked.connect(self.run_program)

        self.view_video = QtWidgets.QTextEdit(self.centralwidget)
        self.view_video.setObjectName("view_video")
        self.gridLayout.addWidget(self.view_video, 12, 1, 1, 1)

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
            self.output_xls.toPlainText(), 
            float(self.fasterrcnn_conf.toPlainText()), 
            int(self.frame_rate.toPlainText())-1, 
            osp.join(self.fasterrcnn_ckpt_file.toPlainText()), 
            self.view_video.toPlainText().lower().capitalize()
        )
        
    def retranslateUi(self, MainWindow):
        """
        Set the text for UI elements.
        
        Args:
            MainWindow (QMainWindow): The main window object.
        """
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt;\">Set Parameters and Click Run Program to Begin</span></p></body></html>"))
        self.fasterrcnn_conf.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">0.31</span></p></body></html>"))
        self.input_vid.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Write Here</span></p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Faster Rcnn Conf Threshold</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">YOLO Conf Threshold: </span></p></body></html>"))
        self.label_12.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">View Video</span></p></body></html>"))
        self.label_11.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Tracker Update Frame Rate</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">YOLO Ckpt File Filepath</span></p></body></html>"))
        self.fasterrcnn_ckpt_file.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Write Here</span></p></body></html>"))
        self.Device.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Write Here</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Faster Rcnn Ckpt File Filepath</span></p></body></html>"))
        self.output_vid.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Write Here</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">IOU Threshold</span></p></body></html>"))
        self.label_9.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Input Video File Filepath</span></p></body></html>"))
        self.frame_rate.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">30</span></p></body></html>"))
        self.label_10.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Output BBox .Xls Filepath</span></p></body></html>"))
        self.yolockpt.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Write Here</span></p></body></html>"))
        self.label_8.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Output Video File Filepath</span></p></body></html>"))
        self.yolo_conf.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">0.33</span></p></body></html>"))
        self.overlap_threshold.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">0.4</span></p></body></html>"))
        self.output_xls.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Write Here</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt;\">Device</span></p></body></html>"))
        self.run_button.setText(_translate("MainWindow", "Run Program"))
        self.view_video.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">False</span></p></body></html>"))






def args_parser(): # recieves arguments from terminal to determine whether to use ui or yaml file
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Training', add_help=True)
    parser.add_argument('--use-yaml', default=False, type=bool, help='if set to True, then yaml file settings will be used instead of ui')
    parser.add_argument('--yaml-path', default='parameters.yaml', type=str, help='path of yaml (--use-yaml argument must be set to True)')
    return parser

def yaml_startup(yaml_filepath):
    # reads in yaml settings and calls main function with these settings
    with open(yaml_filepath, 'r') as file:
        data = yaml.safe_load(file)

    weights = osp.join(data['YOLO_file_location'])
    device = data['device']
    conf_threshold = data['yolo_conf_threshold']
    video_file = data['video_file']
    output_file = data['output_file_location']
    overlap_threshold = data['overlap_threshold']
    xls_file_path = data['xls_filepath']
    rcnn_threshold = data['faster_rcnn_conf_threshold']
    frame_rate = data['frame_rate_of_program']
    faster_rcnn_location = data['faster_rcnn_location']
    view_video = data['view_video']
    main(weights, device, conf_threshold, video_file, output_file, overlap_threshold, xls_file_path, rcnn_threshold, frame_rate, faster_rcnn_location, view_video)



def main(weights, device, conf_threshold, video_file, output_file, iou_threshold, xls_file_path, rcnn_threshold, frame_rate, faster_rcnn_location, view_video):
    """
    Main function to run the detection and tracking process.
    
    Args:
        weights (str): Path to the YOLOv6 weights file.
        device (str): Device to run the models on (e.g., 'cpu' or GPU ID).
        conf_threshold (float): Confidence threshold for YOLOv6.
        video_file (str): Path to the input video file.
        output_file (str): Path to save the output video file.
        iou_threshold (float): IoU threshold for merging boxes.
        xls_file_path (str): Path to save the output XLS file with bounding boxes.
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
    fn_ext = output_file.split(".")[-1].lower() # checking if the video is mp4 or avi
    if fn_ext == 'avi':
        result = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), 30, size)
    elif fn_ext == 'mp4':
        result = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    else:
        print('video file is incompatible with program')
        exit()


    # Define OpenCV object trackers
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.legacy.TrackerCSRT_create,
        "kcf": cv2.legacy.TrackerKCF_create,
        "mil": cv2.legacy.TrackerMIL_create,
    }

    if view_video == True: # initializing window for viewing annotated video frames if option is turned on
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

        if view_video == True: # setting window size for viewing annotated video frames if option is turned on
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

        # Display the output frame if the option to view video is turned on
        if view_video == True:
            cv2.imshow('output', np_frame)
            cv2.waitKey(30)
        
        # Write the frame to the output video file
        
        result.write(np_frame)
        
        print("frame number: ", ind+1) # Displaying which frame number the video is currently proccessing
        timer += 1
        ind += 1


    # Save bounding box coordinates to xls file
    my_df = pd.DataFrame(saved_bboxes, columns=['Current Frame', 'X Bound, Left', 'X Bound, Right', 'Y Bound, Upper', 'Y Bound, Lower'])
    my_df.to_excel(xls_file_path, engine='openpyxl', index=False)

    # Release resources
    cap.release()
    result.release()
    cv2.destroyAllWindows()



args = args_parser().parse_args()
# Reads in arguments to determine whether to use ui or yaml settings for program

if args.use_yaml == False: # Create the application and the main window
    
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()

    # Setup the UI and show the main window
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    # Start the application event loop

    sys.exit(app.exec_())
else: # Use yaml file as input instead
    yaml_startup(args.yaml_path)

