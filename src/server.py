
################################################################################
#  GRPC Server                                   
################################################################################
__author__ = "Adithya Sampath"
__date__  = "May 2019"

import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..","generated"))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..","utils"))
import object_detector_pb2_grpc as grpc_stub
import object_detector_pb2 as msg_stub
import grpc
from concurrent import futures
import time
from tracker import ObjectTracker

# signal to stop interrupts
stop_signal = False

class ObjectTrackingServicer(grpc_stub.ObjectTrackingServicer):
    def __init__(self):
        self.video_url, self.tf_server, self.class_to_detect,self.tracker = None, None, None, None 

    def start_tracking(self, request, context): 
        self.video_url, self.tf_server, self.class_to_detect = request.video_url,request.tf_server,request.class_name
        print("Press 'S' to start tracking\n'D' to reset tracker\n'Q' to quit")
        self.tracker =  ObjectTracker(self.video_url, self.tf_server, self.class_to_detect)
        tracked_BB = self.tracker.track_object()
        output = msg_stub.Output()
        output.response = 0
        for bbox in tracked_BB:
            if bbox is not None:
                output.response = 1
                temp_box = msg_stub.BoundingBox()
                temp_box.X1 = bbox[0]
                temp_box.Y1 = bbox[1]
                temp_box.X2 = bbox[2]
                temp_box.Y2 = bbox[3]
                output.bounding_box.CopyFrom(temp_box)
            yield output

def run_tracker(PORT):
    global stop_signal
    try:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        tracker_instance = ObjectTrackingServicer()
        grpc_stub.add_ObjectTrackingServicer_to_server(tracker_instance, server)
        server.add_insecure_port('[::]:' + SERVER_PORT)
        server.start()
        print("GRPC Server Started & Listening on  ", PORT)
        while not stop_signal:
            time.sleep(10)  #for faster tests
    except KeyboardInterrupt:
        print("Keyboard interrupt - Going to stop the server")
        stop_signal = True
    except:
        print("Unexpected Error!", sys.exc_info()[0], "occured.")
        stop_signal = True
        print("Going to stop the Object Detector Server")
        server.stop(0)

if __name__ == '__main__':
    PORT = '8090'
    if len(sys.argv) > 1:
        PORT = sys.argv[1]
    print("Going to start Object Detector at port ", PORT)
    run_tracker(PORT=PORT)
    print("Exiting Object Detector")




