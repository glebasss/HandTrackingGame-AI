import cv2
import mediapipe as mp
import time
from handtrackingmodule import *

detector = HandDetector()
lazers = Lazers()
game = SoccerGame()
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        print('aaaa')
        break
    img = game.play_game_ball(frame)
    cv2.imshow('video', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
