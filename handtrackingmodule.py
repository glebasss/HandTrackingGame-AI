import cv2
import mediapipe as mp
import torch
import time
import numpy as np


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            model_complexity=0,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms)
        return img

    def get_positions(self, img):

        lmlist = []
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for numhand, hand in enumerate(results.multi_hand_landmarks):
                templist = []
                # [handNum].landmark
                for id, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    templist.append({'id': id, 'x': cx, 'y': cy})
                lmlist.append(templist)
        return lmlist


class Lazers(HandDetector):
    def __init__(self, p1=4, p2=8):
        super().__init__()
        self.p1 = p1
        self.p2 = p2

    def draw_lazers(self, img):
        img_ = img.copy()
        imgRGB = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            listt = self.get_positions(img_)
            for hand in listt:
                if len(hand) > 15:
                    x1, y1 = hand[self.p1]['x'], hand[self.p1]['y']
                    x2, y2 = hand[self.p2]['x'], hand[self.p2]['y']
                    cv2.line(img_, (x1, y1), (x2, y2), (25, 100, 10), 2)
            return img_
        return img_


# class Ball:
#     def __init__(self,):
#         self.src_ball = r'items\SoccerBall.png'
#         self.img_ball = cv2.imread(
#             self.src_ball, cv2.IMREAD_UNCHANGED)
#         self.img_ball = cv2.resize(self.img_ball, (70, 70))
#         self.alpha_channels = self.img_ball[:, :, 3]
#         self.mask = (self.alpha_channels > 0).astype(np.uint8)
#         self.height, self.width, _ = self.img_ball.shape


class SoccerGame:
    def __init__(self,):
        self.lazers = Lazers()
        self.ball_spawned = False

    def ball_spawn(self, img):
        h, w, c = img.shape
        h_max, h_min = (5*h/20), (h*3/20)
        w_max, w_min = (15*w/20), (w*4/20)
        h_max, h_min = int(h_max), int(h_min),
        w_max, w_min = int(w_max), int(w_min)
        y = np.random.randint(h_min, h_max)
        x = np.random.randint(w_min, w_max)
        self.ball_spawned = True
        return x, y

    def falling_ball(self):
        self.yball += 2

    def play_game_ball(self, img):
        img_ = img.copy()
        img_ = self.lazers.draw_lazers(img_)
        if self.ball_spawned == False:
            self.xball, self.yball = self.ball_spawn(img_)
        self.falling_ball()
        cv2.circle(img_, (self.xball, self.yball), 20, (255, 0, 255), -1)
        cv2.circle(img_, (self.xball, self.yball), 21, (0, 0, 0), 2)
        return img_
