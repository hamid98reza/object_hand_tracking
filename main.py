import cv2
import numpy as np
from datetime import datetime
import argparse
from imutils.video import VideoStream
import imutils
import mediapipe as mp
import time

# use python main.py for webcam and  python main.py --video PathOfVideo for videos.

def track():
    a_parse = argparse.ArgumentParser()
    a_parse.add_argument("-v", "--video")
    arguments = vars(a_parse.parse_args())

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius=2)
    pTime = 0


    if not arguments.get("video",False):
        cap = VideoStream(src=0).start()   #IF AN ERROR HAPPENED WHILE
                                            # USING WEBCAM, THIS IS BECAUSE OF 
                                            # IMUTILS , CHECK BALLTRACKING.        
            
    else:
        cap = cv2.VideoCapture(arguments["video"])   

    time.sleep(2.0)
    width = 640
    height = 360

    # Define range of yellow color in HSV
    lower_yellow = np.array([20, 50, 70])
    upper_yellow = np.array([35, 255, 255])
    # lower_yellow = np.array([0, 0, 0])
    # upper_yellow = np.array([360, 300, 30])
    # lower_yellow = np.array([36, 50, 70])
    # upper_yellow = np.array([89, 255, 255])


    start_time = datetime.now()
    x = 0
    while True:
        img = cap.read()
        img = img[1] if arguments.get("video", False) else img
        img = imutils.resize(img,width,height)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = imutils.resize(img,width,height)


        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x*width), int(lm.y*height)
                    # cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

        blurred = cv2.GaussianBlur(img2,(9,9),0)
        # Convert image to HSV color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)


        # Get contours from mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over all identified contours
        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area>5000:

                    # Get the bounding box of the contours
                    x,y,w,h = cv2.boundingRect(cnt)

                    center_width = x+w/2
                    center_height = y+h/2

                    # print(f"{center_width/width} {center_height/height} {w/width} {h/height}")


                    # Print the coordinates of the bounding box

                    # print(f"Top left corner: ({x}, {y})")
                    # print(f"Top right corner: ({x + w}, {y})")
                    # print(f"Bottom left corner: ({x}, {y + h})")
                    # print(f"Bottom right corner: ({x + w}, {y + h})")

                    cv2.putText(img2,f'({x,y})', (x, y-10) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12),2)
                    cv2.putText(img2, f"({x+w,y})", (x+w, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.5 , (36,255,12), 2)
                    cv2.putText(img2, f"({x,y+h})", (x, (y+h)+10), cv2.FONT_HERSHEY_SIMPLEX,0.5 , (36,255,12), 2)
                    cv2.putText(img2, f"({x+w,y+h})", (x+w, (y+h)+10), cv2.FONT_HERSHEY_SIMPLEX,0.5 , (36,255,12), 2)


                    # Draw the bounding box on the original image
                    img2 = cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)



                    x+=1
                    if x%4==0:
                        
                        # cv2.imwrite()               # you can write the image on desired time to a folder to 
                                                                # store the specific time
                        print("...............added")

        # you can have fps here
        cTime = time.time()
        fps = 0/(cTime-pTime)
        pTime = cTime

        cv2.imshow('image2', img2)
        cv2.imshow('image', img)
        cv2.imshow("blurred",mask)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    if not arguments.get("video",False):
        cap.stop()
    else:     
        cap.release()
    
    cv2.destroyAllWindows()

if __name__=="__main__":
     track()
