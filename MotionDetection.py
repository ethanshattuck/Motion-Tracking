import cv2
#import pigpio
import math
import time
from imutils.video import WebcamVideoStream

print("Sampling frames.. The computer will set the initial appearance of the room, then look for differences later")

count = 0
FirstImage = None
setup = True
cap = cv2.VideoCapture(0)

while (setup):
    ret, frame = cap.read()

    if (count < 30):
        print(count)
        count +=1
    else:
        print("background stabilized")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15,15), 0)
        FirstImage = gray
        setup = False
        cap.release()

vs = WebcamVideoStream(src=0).start()
while (True):
    frame = vs.read().copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15,15), 0)
    difference = cv2.absdiff(FirstImage, gray)
    difference = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)[1]
    difference = cv2.GaussianBlur(difference, (15,15), 0)
    difference = cv2.dilate(difference, None, iterations = 2)

    contours, hierarchy = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_area = 2000
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > best_area:
            best_area = area
            best_cnt = cnt

    c = best_cnt

    if c is not None:
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break;

cv2.destroyAllWindows()
vs.stop()
