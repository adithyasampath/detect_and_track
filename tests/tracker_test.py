################################################################################
#  Test cases                                    
################################################################################
__author__ = "Adithya Sampath"
__date__  = "May 2019"

import unittest
from threading import Thread
import time
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..","generated"))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..","config"))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..","src"))
from config import config
import object_detector_pb2_grpc as grpc_stub
import object_detector_pb2 as message_stub
import grpc
import server


class TestObjectDetector(unittest.TestCase):
    def test_start_tracking(self):
        IP_PORT = config['IP_PORT']
        print("Going to start Object detector client for Server at ", IP_PORT)
        channel = grpc.insecure_channel(IP_PORT)
        client = grpc_stub.ObjectTrackingStub(channel)
        video_details = message_stub.Settings()
        video_details.drone_id = config['drone_id']
        video_details.video_url = config['video_url']
        video_details.tf_server = config['tf_server']
        video_details.class_name = config['class_to_detect']
        output_stream = client.start_tracking(video_details)
        for output in output_stream:
                if output.response is not False:
                    print("Detected/Tracked at ", output.bounding_box)

if __name__ == '__main__':
    PORT = config['IP_PORT']
    print("Going to start Object Detector at port ", PORT)
    t = Thread(target=server.run_tracker, args=(PORT,))
    t.start()
    time.sleep(.25)
    unittest.main(verbosity=2)
    print("Exiting Object Detector")
