import cv2
import mediapipe as mp
import torch
import time
import handtrackingmodule as htm


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = htm.handDetector()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        pos = detector.find_position(img, 0)
        print(pos)
        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        cv2.imshow('Image',   img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
