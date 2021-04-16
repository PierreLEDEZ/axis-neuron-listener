import numpy as np
import threading
import time
import cv2
import tensorflow
from tensorflow.keras import models
from .utils import utils
from .bvh import bvh_parser
import lib.bvh_transformation as bvh_transformation

class Predicter:
    def __init__(self):
        self.frames = []
        self.predictions = []
        self.model = models.load_model("./model/classifier.h5")
        self.classes = utils.CLASSES
        self.bvhParser = bvh_parser.BVHParser()
        self.bvhParser.parse("./skeleton.bvh")
        self.joints = self.bvhParser.get_joints_list()
        self.ignored_joints_index = utils.ignoreJoints(self.bvhParser, "geo", utils.IGNORED_JOINTS)
    
    def update_frames(self, frames):
        self.frames = np.array(frames, dtype=np.float32)

    def bvh2RGB(self):
        width, height = len(self.frames), len(self.joints)*2
        img_struct = (height, width, 3)

        img_content = np.zeros(img_struct, dtype=np.uint8)
        
        translation_min_max_X, translation_min_max_Y, translation_min_max_Z = utils.get_min_max_translation(self.frames)
        rotation_min_max_X, rotation_min_max_Y, rotation_min_max_Z = utils.get_min_max_rotation(self.frames)

        for nb_frame, frame in enumerate(self.frames):
            joint_placed = 0
            for nb_joint in range(0, height):
                if nb_joint in self.ignored_joints_index:
                    joint_placed += 3
                    continue
                if nb_joint % 2 == 0:
                    img_content[nb_joint][nb_frame] = utils.calculate_laraba_value_translation(frame, joint_placed, translation_min_max_X, translation_min_max_Y, translation_min_max_Z)
                else:
                    img_content[nb_joint][nb_frame] = utils.calculate_laraba_value_rotation(frame, joint_placed, rotation_min_max_X, rotation_min_max_Y, rotation_min_max_Z)

                joint_placed += 3
        img_content = np.delete(img_content, self.ignored_joints_index, axis=0)

        R,G,B = cv2.split(img_content)
        img_content = cv2.merge([B,G,R])

        final_img = cv2.resize(img_content, (236,118), interpolation=cv2.INTER_NEAREST)

        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2HSV)
        H,S,V = cv2.split(final_img)
        ahe = cv2.createCLAHE(clipLimit=CLIPLIMIT, tileGridSize=(8,8))
        V = ahe.apply(V)
        final_img = cv2.merge([H,S,V])
        final_img = cv2.cvtColor(final_img, cv2.COLOR_HSV2BGR)
        return final_img

    def run(self):
        t = threading.Thread(target=self.predict_classes)
        t.daemon=True
        t.start()

    def predict_classes(self):
        final_img = bvh_transformation.bvh2GeometricFeaturesV2(self.frames, self.joints, utils.IGNORED_JOINTS, self.ignored_joints_index)
        final_img = np.array([final_img], dtype=np.float32)
        # final_img = tensorflow.keras.applications.resnet.preprocess_input(final_img)
        predictions = self.model.predict(final_img)[0]
        self.predictions = predictions