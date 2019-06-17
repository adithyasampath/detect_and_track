################################################################################
#  Helper functions  for image preoricessing                                   #
################################################################################
__author__ = "Adithya Sampath"
__date__ = "May 2019"

from imutils.video import FPS
from timeit import default_timer as timer
import numpy as np
import cv2
from threading import Thread
import sys
import queue


class FileVideoStream:
    def __init__(self, video_url, queueSize=1):
        self.video_stream = cv2.VideoCapture(video_url)
        self.stopped = False
        self.Q = queue.Queue(maxsize=queueSize)
        self.height_scale, self.width_scale = 800, 600
        self.height, self.width = 1920, 1080
        self.scaleX, self.scaleY = 1, 1
        self.fps = FPS()

    def check_scaling(self):
        # scale factor of original vs new frame size
        ok, frame = self.video_stream.read()
        init_size = frame.shape[:2]
        self.height, self.width = init_size[0], init_size[1]
        frame_resize = image_resize(frame, self.width_scale, self.height_scale)
        new_size = frame_resize.shape[:2]
        self.scaleX, self.scaleY = (init_size[0] / new_size[0]), (
            init_size[1] / new_size[1])

    def scale_bbox(self, bbox):
        # re-scale bounding box to original scale
        self.check_scaling()
        bbox[0], bbox[2] = bbox[0] * self.scaleX, bbox[2] * self.scaleX
        bbox[1], bbox[3] = bbox[1] * self.scaleY, bbox[3] * self.scaleY
        return bbox

    def draw_rect(self, frame, bbox):
        # Utility method to draw the detected rectange
        bbox_new = self.scale_bbox(bbox)
        cv2.rectangle(frame, (bbox_new[0], bbox_new[1]),
                      (bbox_new[2], bbox_new[3]), (0, 255, 0), 2)
        return frame

    def visualize_frame(self, frame, bbox, success):
        # Utility method to visualize the frame
        info = [
            ("Tracker", "CSRT"),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(self.fps.fps())),
        ]
        frame = self.draw_rect(frame, bbox)
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text,
                        (self.width - 220, self.height - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)

    def read_frame(self):
        # read frame from video stream, rezise frame and put to queue
        while not self.stopped:
            if not self.Q.full():
                ok, frame = self.video_stream.read()
                self.fps.update()
                if not ok:
                    self.stop()
                    return None
                frame_resize = image_resize(frame, self.width_scale,
                                            self.height_scale)
                self.Q.put(frame_resize, timeout=.033)
            else:
                print("Skipping frame")
                continue

    def start(self):
        # start frame read thread
        t = Thread(target=self.read_frame, args=())
        t.daemon = True
        self.fps.start()
        t.start()
        return self

    def read(self):
        # read frames from queue
        try:
            return self.Q.get(block=False, timeout=.001)
        except queue.Empty:
            print("Queue Empty!!")
            pass

    def stop(self):
        # stops video frame read
        self.stopped = True
        self.fps.stop()
        self.video_stream.release()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0


def decode_image_opencv(image,
                        max_height=800,
                        swapRB=True,
                        imagenet_mean=(0, 0, 0)):
    ### Going to create image vector via OpenCV
    #todo https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/getting%20started.html
    start = timer()
    (h, w) = image.shape[:2]
    image = image_resize(image, height=max_height)
    org = image
    # more details https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    #IMAGENET_MEAN = (103.939, 116.779, 123.68)
    image = cv2.dnn.blobFromImage(
        image, scalefactor=1.0, mean=imagenet_mean, swapRB=swapRB)
    # this gives   shape as  (1, 3, 480, 640))
    image = np.transpose(image, (0, 2, 3, 1))
    # we get it after transpose as ('Input shape=', (1, 480, 640, 3))
    end = timer()
    return image, org


#https://stackoverflow.com/a/44659589/429476
# It is important to resize without loosing the aspect ratio for
# good detection
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


################################################################################
