import cv2
import mediapipe as mp
import torch
import time
import numpy as np
from utils import Text


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
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            listt = self.get_positions(img)
            for hand in listt:
                if len(hand) > 15:
                    x1, y1 = hand[self.p1]['x'], hand[self.p1]['y']
                    x2, y2 = hand[self.p2]['x'], hand[self.p2]['y']
                    cv2.line(img, (x1, y1), (x2, y2), (25, 100, 10), 2)
            return img
        return img

    def lazer_points(self, img):
        img_ = img.copy()
        imgRGB = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            listt = self.get_positions(img_)
            for hand in listt:
                if len(hand) > 15:
                    x1, y1 = hand[self.p1]['x'], hand[self.p1]['y']
                    x2, y2 = hand[self.p2]['x'], hand[self.p2]['y']
                    return x1, y1, x2, y2
        return None

    def get_points_between(self, x1, y1, x2, y2):
        points = []
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))

        x_increment = dx / steps
        y_increment = dy / steps

        x, y = x1, y1

        for i in range(steps + 1):
            points.append((int(x), int(y)))
            x += x_increment
            y += y_increment

        return points


class FingerDot(HandDetector):
    def __init__(self, point):
        super().__init__()
        self.point = point

    def dot_points(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            listt = self.get_positions(img)
            for hand in listt:
                if len(hand) > 15:
                    x1, y1 = hand[self.point]['x'], hand[self.point]['y']
                    return x1, y1

    def dot_show(self, img):
        listt = self.dot_points(img)
        if listt != None:
            x1, y1 = listt
            cv2.circle(img, (x1, y1), 15, (0, 255, 0), -1)


class SoccerGame:
    def __init__(self,):
        self.lazers = Lazers()
        self.ball_spawned = False
        self.start_num = -1
        self.score = 0
        self.end = 0
        self.text_start = Text(cv2.FONT_HERSHEY_SIMPLEX, 3, 5)
        self.text_end = Text(cv2.FONT_HERSHEY_SIMPLEX,
                             3, 5, color=(0, 255, 255))
        self.fingerdot = FingerDot(8)

    def ball_spawn(self, img):
        self.h, self.w, _ = img.shape
        h_max, h_min = (5*self.h/20), (self.h*3/20)
        w_max, w_min = (15*self.w/20), (self.w*4/20)
        h_max, h_min = int(h_max), int(h_min)
        w_max, w_min = int(w_max), int(w_min)
        self.yball = int(np.random.randint(h_min, h_max))
        self.xball = int(np.random.randint(w_min, w_max))
        self.ball_spawned = True

    def falling_ball(self):
        self.ky = (1+(pow(self.score, (14/10))/10))
        self.yball += 2 * int(self.ky)

    def is_ball_touch(self, xball, yball, radius, point, detection='lazer'):
        if detection == 'lazer':
            distance = ((point[0] - xball) ** 2 +
                        (point[1] - yball) ** 2) ** 0.5
            return distance <= radius
        if detection == 'fingerdot':
            distance = np.sqrt((self.x1dot - xball) **
                               2 + (self.y1dot - yball) ** 2)
            return distance < 35

    def collision(self, img, detection):
        if detection == 'lazer':
            points = self.lazers.lazer_points(img)
            if points != None:
                x1, y1, x2, y2 = points
                list_of_points = self.lazers.get_points_between(x1, y1, x2, y2)
                for point in list_of_points:
                    if self.is_ball_touch(self.xball, self.yball, 20, point):
                        self.ball_spawn(img)
                        self.score += 1
        if detection == 'fingerdot':
            points = self.fingerdot.dot_points(img)
            if points != None:
                self.x1dot, self.y1dot = points
                if self.is_ball_touch(self.xball, self.yball, 20, points, 'fingerdot'):
                    self.ball_spawn(img)
                    self.score += 1

    def start_game(self, img):
        match self.start_num:
            case 0:
                self.text_start.put_in_center("Ready?", img)
            case 1:
                self.text_start.put_in_center('3', img)
            case 2:
                self.text_start.put_in_center('2', img)
            case 3:
                self.text_start.put_in_center('1', img)
        return img

    def end_game(self, img):
        end_text = f'LOSER!!!'
        self.text_end.put_in_center(end_text, img)

    def show_score(self, img):
        scoretext = f'Score : {self.score}'
        cv2.putText(img, scoretext, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

    def play_game_ball(self, cap, detection):
        '''
        detection = [lazer,fingerdot]
        '''
        self.display_time = 1
        self.start_time = 0
        while True:
            ret, img = cap.read()
            if not ret:
                print('no ret')
                break
            img = cv2.flip(img, 1).copy()
            if self.start_num <= 3:
                self.start_game(img)
                if time.time()-self.start_time > self.display_time:
                    self.start_num += 1
                    self.start_time = time.time()
            else:
                if self.ball_spawned == False:
                    self.ball_spawn(img)
                if self.yball > (self.h+5):
                    if self.end == 0:
                        end_time = time.time()
                        self.end = 1
                    self.end_game(img)
                    if time.time() - end_time >= 3:
                        self.end = 2
                else:
                    if detection == 'lazer':
                        self.lazers.draw_lazers(img)
                    if detection == 'fingerdot':
                        self.fingerdot.dot_show(img)
                    self.falling_ball()
                    self.collision(img, detection=detection)
                    self.show_score(img)
                    cv2.circle(img, (self.xball, int(self.yball)),
                               20, (255, 0, 255), -1)
                    cv2.circle(img, (self.xball, int(self.yball)),
                               21, (0, 0, 0), 2)
            cv2.imshow('video', img)
            if cv2.waitKey(10) & 0xFF == ord('q') or (self.end == 2):
                cap.release()
                cv2.destroyAllWindows()
                break
