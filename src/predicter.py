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

classes = ["consult_sheets", "picking_in_front", "picking_left", "take_screwdriver"]

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
        self.model = models.load_model("./model/MISO_resnet50_mkI_1_5_9_4classes_average_1dense6144_1denseclassification.h5")
        self.classes = utils.CLASSES
        self.bvhParser = bvh_parser.BVHParser()
        self.bvhParser.parse("./skeleton.bvh")
        self.joints = self.bvhParser.get_joints_list()
        self.ignored_joints_index = utils.ignoreJoints(self.bvhParser, "geo", utils.IGNORED_JOINTS)
        self.killed = False
        self.mode = mode
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

        # final_img = bvh_transformation.bvh2GeometricFeaturesV2(self.frames, self.joints, utils.IGNORED_JOINTS, self.ignored_joints_index)
        # final_img = tensorflow.keras.applications.resnet.preprocess_input(final_img)
        
        # ==========================================================
        
        # joints_we_want = np.array([0, 13, 16, 17, 20], dtype=np.uint8)
        # final_img = bvh_transformation.bvh2GeometricFeaturesCustom(self.frames, self.joints, utils.IGNORED_JOINTS, self.ignored_joints_index, joints_we_want)
        # final_img = np.array([final_img], dtype=np.float32)
        # final_img = tensorflow.keras.applications.resnet.preprocess_input(final_img)
        
        # predictions = self.model.predict()

        # ==========================================================
        
        final_images = bvh_transformation.bvh2MultipleImages(self.frames, self.joints, utils.IGNORED_JOINTS, self.ignored_joints_index)
        for i in range(len(final_images)):
            B,G,R = cv2.split(final_images[i])
            final_images[i] = cv2.merge([R,G,B])
        predictions = self.model.predict([np.array([final_images[0]]), np.array([final_images[1]]), np.array([final_images[2]])])
        self.predictions = predictions[0]
        if self.mode != "web":
            if self.activity_analyzer.started:
                predicted_class_index = np.argmax(predictions[0], axis=-1)
                self.predicted_action.put(classes[predicted_class_index])
        # return
        # results_0 = predictions[0][0]
        # results_1 = predictions[1][0]
        # results_2 = predictions[2][0]
        # final_preds = []
        # for pred_0, pred_1, pred_2 in zip(results_0, results_1, results_2):
        #     final_pred = 1/3*sum([pred_0, pred_1, pred_2])
        #     final_preds.append(final_pred)
        # print("=================================================================================================================\n{}\n{}\n{}\n---------------------------------------------------------------\n{}\n=================================================================================================================\n\n\n".format((results_0*100).tolist(), (results_1*100).tolist(), (results_2*100).tolist(), final_preds))
        # print(final_preds)
        
        # ===========================================================
        
        # final_images = bvh_transformation.bvh2MultipleImages(self.frames, self.joints, utils.IGNORED_JOINTS, self.ignored_joints_index)
        # predictions = self.model.predict([np.array([final_images[0]]), np.array([final_images[1]])])
        # results_0 = predictions[0][0]
        # results_1 = predictions[1][0]
        # final_preds = []
        # for pred_0, pred_1 in zip(results_0, results_1):
        #     final_pred = 1/4*sum([pred_0, pred_1])
        #     final_preds.append(final_pred)
        # print(final_preds)