import cv2
import mediapipe as mp
import time
import numpy as np
from utils import Text, Rectangle

# Class to detect hand landmarks using MediaPipe


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.8):
        # Initialize MediaPipe Hands with specified parameters
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
        # Convert image to RGB and process with MediaPipe Hands
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        # Draw hand landmarks if any detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms)
        return img

    def get_positions(self, img):
        # Get coordinates of all landmarks in detected hands
        lmlist = []
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for numhand, hand in enumerate(results.multi_hand_landmarks):
                templist = []
                for id, lm in enumerate(hand.landmark):
                    # Convert landmark coordinates to image pixel space
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    templist.append({'id': id, 'x': cx, 'y': cy})
                lmlist.append(templist)
        return lmlist

# Class to draw lasers between two finger points


class Lazers(HandDetector):
    def __init__(self, p1=4, p2=8):
        super().__init__()
        self.p1 = p1  # First point of laser (index of landmark)
        self.p2 = p2  # Second point of laser (index of landmark)

    def draw_lazers(self, img):
        # Draw a laser between points p1 and p2 on the detected hand
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
        # Get the coordinates of the laser endpoints
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
        # Calculate points between two given points (for laser line)
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

# Class to track a specific point (dot) on the finger


class FingerDot(HandDetector):
    def __init__(self, point):
        super().__init__()
        self.point = point  # Landmark index to track

    def dot_points(self, img):
        # Get the position of the tracked point
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
        # Draw a dot on the tracked point
        listt = self.dot_points(img)
        if listt != 0:
            for hand in listt:
                x1, y1 = hand
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), -1)

# Main game class to implement a game involving lasers and hand detection


class SoccerGame:
    def __init__(self):
        self.lazers = Lazers()
        self.ball_spawned = False
        self.start_num = -1
        self.score = 0
        self.end = 0
        # Text utility classes for displaying messages
        self.text_start = Text(cv2.FONT_HERSHEY_SIMPLEX, 3, 5)
        self.text_end = Text(cv2.FONT_HERSHEY_SIMPLEX,
                             1, 3, color=(0, 255, 255))
        self.fingerdot = FingerDot(8)  # Track fingertip (landmark 8)
        self.selected_mod = 0
        self.rectange = Rectangle()
        # Difficulty levels for the game
        self.easy_count = [0, 'easy', 0]
        self.medium_count = [0, 'medium', 0]
        self.hard_count = [0, 'hard', 0]

    def ball_spawn(self, img):
        # Randomly spawn a ball in a specific range within the image
        self.h, self.w, _ = img.shape
        h_max, h_min = (5*self.h/20), (self.h*3/20)
        w_max, w_min = (15*self.w/20), (self.w*4/20)
        h_max, h_min = int(h_max), int(h_min)
        w_max, w_min = int(w_max), int(w_min)
        self.yball = int(np.random.randint(h_min, h_max))
        self.xball = int(np.random.randint(w_min, w_max))
        self.ball_spawned = True

    def falling_ball(self):
        # Update the position of the falling ball, speed increases with score
        self.ky = (1+(pow(self.score, (14/10))/10)) * self.coef_falling
        self.yball += 2 * int(self.ky) * self.coef_falling

    def is_ball_touch(self, xball, yball, radius, point, detection='lazer'):
        # Check if the ball is touched by the laser or fingertip
        if detection == 'lazer':
            distance = ((point[0] - xball) ** 2 +
                        (point[1] - yball) ** 2) ** 0.5
            return distance <= radius
        if detection == 'fingerdot':
            distance = np.sqrt((self.x1dot - xball) ** 2 +
                               (self.y1dot - yball) ** 2)
            return distance < 35

    def collision(self, img, detection):
        # Check for collision between ball and detection method (laser or fingertip)
        if detection == 'lazer':
            # Get the coordinates of laser lines
            hand_points = self.lazers.lazer_points(img)
            if hand_points == None:
                return 0
            for points in hand_points:
                x1, y1, x2, y2 = points
                # Get all points along the laser line
                list_of_points = self.lazers.get_points_between(x1, y1, x2, y2)
                # Check if the ball is touched by any point along the laser line
                for point in list_of_points:
                    if self.is_ball_touch(self.xball, self.yball, 20, point):
                        self.ball_spawn(img)  # Respawn the ball if touched
                        self.score += 1  # Increase score
        if detection == 'fingerdot':
            # Get fingertip positions
            hand_points = self.fingerdot.dot_points(img)
            if hand_points == 0:
                return 0
            for point in hand_points:
                self.x1dot, self.y1dot = point
                # Check if ball is touched by fingertip
                if self.is_ball_touch(self.xball, self.yball, 20, point, 'fingerdot'):
                    self.ball_spawn(img)
                    self.score += 1

    def rectangle_collision(self, dot, rectangle, mode):
        '''
        Check if a dot (fingertip) is inside a rectangle (selection box for game modes)

        dot : [x, y] (coordinates of the point)
        rectangle : [x1, y1, x2, y2] (coordinates of rectangle corners)
        '''
        count, game_mode, finish = mode
        # If point remains inside the rectangle for a certain time, selection is confirmed
        if count >= 10:
            self.game_mode = game_mode
            return [count, game_mode, 1]
        if not dot:
            count = 0
            return [count, game_mode, 0]
        x_dot, y_dot = dot
        x1, y1, x2, y2 = rectangle
        # Check if the dot is within the rectangle's boundaries
        if (x_dot in range(min(x1, x2), max(x1, x2))) and \
                (y_dot in range(min(y1, y2), max(y1, y2))):
            return [count+1, game_mode, 0]
        count = 0
        return [count, game_mode, 0]

    def start_game(self, img):
        # Display countdown messages before game starts
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
        # Display end game message and the final score
        end_text = f'Nice try!'
        total_score = f'Your total score: {self.score} '
        self.text_end.put_in_center([end_text, total_score], img)

    def show_score(self, img):
        # Display current score on the screen
        scoretext = f'Score : {self.score}'
        cv2.putText(img, scoretext, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

    def select_game_mod(self, cap):
        # User selects the difficulty level of the game (easy, medium, hard)
        while True:
            ret, img = cap.read()
            img = cv2.flip(img, 1).copy()
            if not ret:
                print('no ret')
            dot_points = self.fingerdot.dot_points(img)
            # Display game mode selection text
            self.text_select_mod = Text(
                cv2.FONT_HERSHEY_SIMPLEX, 1, 3, (0, 0, 0))
            _, text_pos = self.text_select_mod.put_in_center(
                ['Select game mod', 'easy', 'medium', 'hard'], img)
            _, self.easy, self.medium, self.hard = text_pos
            # Draw selection rectangles around each game mode option
            self.rectangle_e = self.rectange.create_rectangle(self.easy, img)
            self.rectangle_e = self.rectange.create_rectangle(self.medium, img)
            self.rectangle_e = self.rectange.create_rectangle(self.hard, img)
            self.text_select_mod.put_in_center(
                ['Select game mod', 'easy', 'medium', 'hard'], img)
            # Detect fingertip inside selection boxes to make a selection
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
            # Confirm game mode selection and set difficulty level
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
        # User selects the detection method (laser or fingertip)
        self.game_mode_lazer_count = [0, 'lazer', 0]
        self.game_mode_fingerdot_count = [0, 'fingerdot', 0]
        while True:
            ret, img = cap.read()
            img = cv2.flip(img, 1).copy()
            if not ret:
                print('no ret')
            dot_points = self.fingerdot.dot_points(img)
            # Display text for detection method selection
            self.text_select_detection = Text(
                cv2.FONT_HERSHEY_SIMPLEX, 1, 3, (0, 0, 0))
            _, text_pos = self.text_select_mod.put_in_center(
                ['Select detection', 'lazer', 'fingerdot'], img)
            _, self.game_mode_lazer, self.game_mode_fingerdot = text_pos
            # Draw selection rectangles for laser and fingertip options
            self.rectangle_e = self.rectange.create_rectangle(
                self.game_mode_lazer, img)
            self.rectangle_e = self.rectange.create_rectangle(
                self.game_mode_fingerdot, img)
            self.text_select_detection.put_in_center(
                ['Select detection', 'lazer', 'fingerdot'], img)
            # Detect fingertip inside selection boxes to make a selection
            if dot_points:
                dot_point = dot_points[0]
                cv2.circle(
                    img, (dot_point[0], dot_point[1]), 15, (0, 255, 50), -1)
                self.game_mode_lazer_count = self.rectangle_collision(
                    dot_point, self.game_mode_lazer, self.game_mode_lazer_count)
                self.game_mode_fingerdot_count = self.rectangle_collision(
                    dot_point, self.game_mode_fingerdot, self.game_mode_fingerdot_count)
            # Confirm detection mode selection
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
        # Main loop for the game involving the falling ball
        self.display_time = 1  # Time delay for starting message display
        self.start_time = 0
        # User selects game mode and detection method
        self.select_game_mod(cap)
        self.select_detection(cap)
        while True:
            ret, img = cap.read()
            if not ret:
                print('no ret')
                break
            img = cv2.flip(img, 1).copy()
            # Countdown before the game starts
            if self.start_num <= 3:
                self.start_game(img)
                if time.time() - self.start_time > self.display_time:
                    self.start_num += 1
                    self.start_time = time.time()
            else:
                # Spawn the ball if not spawned already
                if self.ball_spawned == False:
                    self.ball_spawn(img)
                # If the ball falls out of the screen, end the game
                if self.yball > (self.h + 5):
                    if self.end == 0:
                        end_time = time.time()
                        self.end = 1
                    self.end_game(img)
                    if time.time() - end_time >= 3:
                        self.end = 2
                else:
                    # Draw the detection method (laser or fingertip)
                    if self.detection == 'lazer':
                        self.lazers.draw_lazers(img)
                    if self.detection == 'fingerdot':
                        self.fingerdot.dot_show(img)
                    # Update ball position
                    self.falling_ball()
                    # Check for collisions between detection and ball
                    self.collision(img, detection=self.detection)
                    # Display the current score
                    self.show_score(img)
                    # Draw the ball on the screen
                    cv2.circle(img, (self.xball, int(self.yball)),
                               20, (255, 0, 255), -1)
                    cv2.circle(img, (self.xball, int(self.yball)),
                               21, (0, 0, 0), 2)

            # Show the current frame
            cv2.imshow('video', img)
            # Quit the game if 'q' is pressed or game ends
            if cv2.waitKey(10) & 0xFF == ord('q') or (self.end == 2):
                cap.release()
                cv2.destroyAllWindows()
                break
