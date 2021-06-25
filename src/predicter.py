import numpy as np
import threading
import queue
import time
import cv2
import tensorflow
from tensorflow.keras import models
from .utils import utils
from .bvh import bvh_parser
import lib.bvh_transformation as bvh_transformation
from .activity_analyzer.analyzer import ActivityAnalyzer

activities_json = { 
    "activityA": [
        "picking_in_front", 
        "picking_left", 
        "turn_sheets",
        "take_screwdriver",
        "picking_left"
    ],
    "activityB": [
        "picking_left",
        "picking_in_front",
        "take_screwdriver",
        "assemble_system",
        "turn_sheets"
    ],
    "activityC": [
        "picking_in_front",
        "picking_left",
        "turn_sheets",
        "consult_sheets",
        "picking_left"
    ]
}

class Predicter:
    """
        Class in charge of converting frames in image(s) and make predictions.
    """

    def __init__(self, mode):
        """
            Initialize the predicter

            Load the pretrained model, parse an example bvh file to obtain the skeleton
            Get the joints list and the joints list of ignored joints
        """

        self.frames = []
        self.predictions = []
        self.model = models.load_model("./model/MISO_alexnet_1_2_5_8_9_energy1_4classes_average_1dense4096_1denseclassification.h5")
        # self.model = models.load_model("./model/MISO_alexnet_1_2_5_7_9_energy_08_4classes_inhard.h5")
        self.classes = utils.CLASSES
        self.bvhParser = bvh_parser.BVHParser()
        self.bvhParser.parse("./skeleton.bvh")
        self.joints = self.bvhParser.get_joints_list()
        self.ignored_joints_index = utils.ignoreJoints(self.bvhParser, "geo", utils.IGNORED_JOINTS)
        self.killed = False
        self.mode = mode
        self.threshold = 0.5
        if self.mode != "web":
            self.predicted_action = queue.Queue()
            self.activity_analyzer = ActivityAnalyzer(activities_json, self.predicted_action, self.mode)
            self.activity_analyzer.start()
    
    def update_frames(self, frames):
        """
            Save the received frames in a numpy array

            :param frames: received frames from the client thread
            :type frames: list
        """

        self.frames = np.array(frames, dtype=np.float32)

    def kill(self):
        """ Break the activity analyzer loop when called """
        if self.mode != "web":
            self.activity_analyzer.killed = True

    def make_first_prediction(self):
        """
        """

        images_to_use = [1, 2, 5, 7, 9]
        images = []
        
        for i in images_to_use:
            img = cv2.imread("./img/first_prediction/{}.png".format(i))
            images.append(img)

        predictions = self.model([
            np.array([images[0]]),
            np.array([images[1]]),
            np.array([images[2]]),
            np.array([images[3]]),
            np.array([images[4]])
        ])


    def run(self):
        """
            Main function of this class

            Create a thread and start it
        """

        t = threading.Thread(target=self.predict_classes)
        t.daemon=True
        t.start()

    def predict_classes(self):
        """
            Convert received frames in images with the specified method and make predictions
        """

        # final_images = bvh_transformation.bvh2MultipleImages(self.frames, self.joints, utils.IGNORED_JOINTS, self.ignored_joints_index, float(1.0), [1, 2, 5, 8, 9])
        final_images = bvh_transformation.convert_frames_to_imgs(self.frames, self.joints, utils.IGNORED_JOINTS, self.ignored_joints_index, [1, 2, 5, 8, 9])
        
        if final_images != []:
            for i in range(len(final_images)):
                B,G,R = cv2.split(final_images[i])
                final_images[i] = cv2.merge([R,G,B])
            predictions = self.model([
                np.array([final_images[0]]), 
                np.array([final_images[1]]), 
                np.array([final_images[2]]), 
                np.array([final_images[3]]), 
                np.array([final_images[4]])
            ])
            if (np.max(predictions[0], axis=-1) < self.threshold):
                self.predictions = []
                return
            self.predictions = predictions[0]
            if self.mode != "web":
                if self.activity_analyzer.started:
                    predicted_class_index = np.argmax(predictions[0], axis=-1)
                    self.predicted_action.put(utils.CLASSES[predicted_class_index])
