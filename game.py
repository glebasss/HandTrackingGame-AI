import cv2
import mediapipe as mp
import torch
import time
import numpy as np
from utils import Text, Rectangle


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.8):
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
        hands = []
        imgRGB = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            listt = self.get_positions(img_)
            for hand in listt:
                if len(hand) > 15:
                    x1, y1 = hand[self.p1]['x'], hand[self.p1]['y']
                    x2, y2 = hand[self.p2]['x'], hand[self.p2]['y']
                    hands.append([x1, y1, x2, y2])
            return hands
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
        hands = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            listt = self.get_positions(img)
            for hand in listt:
                if len(hand) > 15:
                    x1, y1 = hand[self.point]['x'], hand[self.point]['y']
                    hands.append([x1, y1])
            return hands
        return 0

    def dot_show(self, img):
        listt = self.dot_points(img)
        if listt != 0:
            for hand in listt:
                x1, y1 = hand
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
                             1, 3, color=(0, 255, 255))
        self.fingerdot = FingerDot(8)
        self.selected_mod = 0
        self.rectange = Rectangle()
        self.easy_count = [0, 'easy', 0]
        self.medium_count = [0, 'medium', 0]
        self.hard_count = [0, 'hard', 0]

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
        self.ky = (1+(pow(self.score, (14/10))/10)) * self.coef_falling
        self.yball += 2 * int(self.ky) * self.coef_falling

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
            hand_points = self.lazers.lazer_points(img)
            if hand_points == None:
                return 0
            for points in hand_points:
                x1, y1, x2, y2 = points
                list_of_points = \
                    self.lazers.get_points_between(x1, y1, x2, y2)
                for point in list_of_points:
                    if self.is_ball_touch(self.xball, self.yball, 20, point):
                        self.ball_spawn(img)
                        self.score += 1
        if detection == 'fingerdot':
            hand_points = self.fingerdot.dot_points(img)
            if hand_points == 0:
                return 0
            for point in hand_points:
                self.x1dot, self.y1dot = point
                if self.is_ball_touch(self.xball, self.yball, 20, point, 'fingerdot'):
                    self.ball_spawn(img)
                    self.score += 1

    def rectangle_collision(self, dot, rectangle, mode):
        '''
        dot : [x,y]
        rectangle:[x1,y1,x2,y2]
        '''
        count, game_mode, finish = mode
        if count >= 10:
            self.game_mode = game_mode
            return [count, game_mode, 1]
        if not dot:
            count = 0
            return [count, game_mode, 0]
        x_dot, y_dot = dot
        x1, y1, x2, y2 = rectangle
        if (x_dot in range(min(x1, x2), max(x1, x2))) and \
                (y_dot in range(min(y1, y2), max(y1, y2))):
            return [count+1, game_mode, 0]
        count = 0
        return [count, game_mode, 0]

    def start_game(self, img):
        match self.start_num:
            case 0:
                self.text_start.put_in_center(["Ready?"], img)
            case 1:
                self.text_start.put_in_center(['3'], img)
            case 2:
                self.text_start.put_in_center(['2'], img)
            case 3:
                self.text_start.put_in_center(['1'], img)
        return img

    def end_game(self, img):
        end_text = f'Nice try!'
        total_score = f'Your total score: {self.score} '
        self.text_end.put_in_center([end_text, total_score], img)

    def show_score(self, img):
        scoretext = f'Score : {self.score}'
        cv2.putText(img, scoretext, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

    def select_game_mod(self, cap):
        while True:
            ret, img = cap.read()
            img = cv2.flip(img, 1).copy()
            if not ret:
                print('no ret')
            dot_points = self.fingerdot.dot_points(img)
            self.text_select_mod = Text(
                cv2.FONT_HERSHEY_SIMPLEX, 1, 3, (0, 0, 0))
            _, text_pos = \
                self.text_select_mod.put_in_center(
                    ['Select game mod', 'easy', 'medium', 'hard'], img)
            _, self.easy, self.medium, self.hard = text_pos
            # print(self.easy, self.medium, self.hard)
            self.rectangle_e = self.rectange.create_rectangle(self.easy, img)
            self.rectangle_e = self.rectange.create_rectangle(self.medium, img)
            self.rectangle_e = self.rectange.create_rectangle(self.hard, img)
            self.text_select_mod.put_in_center(
                ['Select game mod', 'easy', 'medium', 'hard'], img)
            if dot_points:
                dot_point = dot_points[0]
                cv2.circle(
                    img, (dot_point[0], dot_point[1]), 15, (0, 255, 50), -1)
                self.easy_count = self.rectangle_collision(
                    dot_point, self.easy, self.easy_count)
                self.medium_count = self.rectangle_collision(
                    dot_point, self.medium, self.medium_count)
                self.hard_count = self.rectangle_collision(
                    dot_point, self.hard, self.hard_count)
            if self.easy_count[2] == 1:
                print(self.game_mode)
                self.coef_falling = 1
                break
            if self.medium_count[2] == 1:
                print(self.game_mode)
                self.coef_falling = 2
                break
            if self.hard_count[2] == 1:
                print(self.game_mode)
                self.coef_falling = 3
                break

            cv2.imshow('video', img)
            if cv2.waitKey(10) & 0xFF == ord('q') or (self.end == 2):
                cap.release()
                cv2.destroyAllWindows()
                break

    def select_detection(self, cap):
        self.game_mode_lazer_count = [0, 'lazer', 0]
        self.game_mode_fingerdot_count = [0, 'fingerdot', 0]
        while True:
            ret, img = cap.read()
            img = cv2.flip(img, 1).copy()
            if not ret:
                print('no ret')
            dot_points = self.fingerdot.dot_points(img)
            self.text_select_detection = Text(
                cv2.FONT_HERSHEY_SIMPLEX, 1, 3, (0, 0, 0))
            _, text_pos = \
                self.text_select_mod.put_in_center(
                    ['Select detection', 'lazer', 'fingerdot'], img)
            _, self.game_mode_lazer, self.game_mode_fingerdot = text_pos
            self.rectangle_e = \
                self.rectange.create_rectangle(self.game_mode_lazer, img)
            self.rectangle_e = \
                self.rectange.create_rectangle(self.game_mode_fingerdot, img)
            self.text_select_detection.put_in_center(
                ['Select detection', 'lazer', 'fingerdot'], img)
            if dot_points:
                dot_point = dot_points[0]
                cv2.circle(
                    img, (dot_point[0], dot_point[1]), 15, (0, 255, 50), -1)
                self.game_mode_lazer_count = self.rectangle_collision(
                    dot_point, self.game_mode_lazer, self.game_mode_lazer_count)
                self.game_mode_fingerdot_count = self.rectangle_collision(
                    dot_point, self.game_mode_fingerdot, self.game_mode_fingerdot_count)
            if self.game_mode_lazer_count[2] == 1:
                self.detection = 'lazer'
                break
            if self.game_mode_fingerdot_count[2] == 1:
                self.detection = 'fingerdot'
                break
            cv2.imshow('video', img)
            if cv2.waitKey(10) & 0xFF == ord('q') or (self.end == 2):
                cap.release()
                cv2.destroyAllWindows()
                break

    def play_game_ball(self, cap):
        self.display_time = 1
        self.start_time = 0
        self.select_game_mod(cap)
        self.select_detection(cap)
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
                    if self.detection == 'lazer':
                        self.lazers.draw_lazers(img)
                    if self.detection == 'fingerdot':
                        self.fingerdot.dot_show(img)
                    self.falling_ball()
                    self.collision(img, detection=self.detection)
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
