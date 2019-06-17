################################################################################
#  Tracker Class
################################################################################
__author__ = "Adithya Sampath"
__date__ = "May 2019"

from threading import Thread
from queue import Queue
import argparse
import imutils
import time
import cv2
import numpy as np
from retinanet_client import Predictor
from image_preprocessing import image_resize, FileVideoStream


class ObjectTracker:
    def __init__(self, video_url, tf_server, class_to_detect):
        self.video_url = video_url
        self.tf_server = tf_server
        self.tracker = cv2.TrackerCSRT_create()
        self.initial_BB = None
        self.object_detector = Predictor()
        self.start_tracking = False
        self.class_to_detect = class_to_detect
        self.video_object = FileVideoStream(video_url=self.video_url)

    def detect_object(self, frame, threshold=0.5):
        # detect object to track using retinanet tensorflow serving model
        boxes, scores, labels = self.object_detector.detect_image(frame)
        for box, score, label in zip(boxes, scores, labels):
            if score < threshold:
                break
            bbox = box.astype(int)
            detected_label = self.object_detector.get_label(int(label))
            if self.class_to_detect in detected_label:
                print("Detected object: ", detected_label,
                      "\nBounding box info: ", bbox, "\nScore: ", score)
                return bbox

    def track_object(self):
        # track object using CSRT tracker
        self.video_object.start()
        while self.video_object.more():
            # read frame from video stream
            frame = self.video_object.read()
            # detect object to track and start tracking
            if self.start_tracking and self.initial_BB is None:
                self.initial_BB = self.detect_object(frame)
                self.tracker.init(frame, self.initial_BB)
            # update and continue tracking
            if self.start_tracking and self.initial_BB is not None:
                success, tracked_bbox = self.tracker.update(frame)
                self.video_object.visualize_frame(frame, tracked_bbox, success)
                if not success:
                    print("Object lost! Detecting again!!")
                    self.initial_BB, tracked_bbox = None, None
            # keys to start and stop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                self.start_tracking = True
            if key == ord("d"):
                self.start_tracking = False
                tracked_bbox, self.initial_BB = None, None
            elif key == ord("q"):
                self.video_object.stop()
                break
            yield tracked_bbox
