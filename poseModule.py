import cv2
import mediapipe as mp
import sys
import numpy as np


class poseDetector:
    def __init__(self, static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.1,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence,
                                     self.min_tracking_confidence)

    def findPose(self, img, img_to_draw, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img_to_draw, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img_to_draw

    def findPosition(self, img, draw=True):
        h, w, _ = img.shape
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                if id <= 10:
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    cv2.circle(img, (cx, cy), 10, (0, 0, 0), cv2.FILLED)
                else:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 4, (255, 255, 255), cv2.FILLED)
        return img


# returns list of images after media pipe process
def getPose(path, isVideo):
    detector = poseDetector()
    cap = cv2.VideoCapture(path)
    pose_list = []
    while True:
        success, img = cap.read()
        if img is None:
            return pose_list
        mask = np.ones(img.shape, np.uint8)
        mask = detector.findPose(img, mask)
        mask = detector.findPosition(mask, draw=True)
        pose_list.append(mask)
