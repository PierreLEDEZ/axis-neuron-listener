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
        self.model = models.load_model("./model/parallel_resnet50_mkI_xy_xz_yz_4classes_without_P01.h5")
        # self.model = models.load_model("./model/parallel_resnet50_mkI_xy_xz_4classes.h5")
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
        results_0 = predictions[0][0]
        results_1 = predictions[1][0]
        results_2 = predictions[2][0]
        final_preds = []
        for pred_0, pred_1, pred_2 in zip(results_0, results_1, results_2):
            final_pred = 1/3*sum([pred_0, pred_1, pred_2])
            final_preds.append(final_pred)
        print("=================================================================================================================\n{}\n{}\n{}\n---------------------------------------------------------------\n{}\n=================================================================================================================\n\n\n".format((results_0*100).tolist(), (results_1*100).tolist(), (results_2*100).tolist(), final_preds))
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


        self.predictions = np.array(final_preds)