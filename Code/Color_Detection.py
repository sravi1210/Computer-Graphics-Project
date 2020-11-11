import numpy as np
import cv2 as cv

def nothing(x):
    pass

# Use webcam to capture the motion of a object
cap = cv.VideoCapture(0)

cv.namedWindow('Tracking')
# Use the trackers to adjust the hue, saturation and value limits
# to tack particular color range ojects. Adjusting the trackers
# the HSV values and hence the color range that is to be tracked
cv.createTrackbar('LH','Tracking',0,255,nothing)
cv.createTrackbar('LS','Tracking',0,255,nothing)
cv.createTrackbar('LV','Tracking',0,255,nothing)
cv.createTrackbar('UH','Tracking',255,255,nothing)
cv.createTrackbar('US','Tracking',255,255,nothing)
cv.createTrackbar('UV','Tracking',255,255,nothing)

while True:
    # Read frame at each time step
    _,frame = cap.read()
    #frame = cv.imread('data/smarties.png',1)

    # convert the frame to HSV format ( Hue, Saturation, Value)
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    # get the lower limit of hsv from trackbar
    lh = cv.getTrackbarPos('LH','Tracking')
    ls = cv.getTrackbarPos('LS','Tracking')
    lv = cv.getTrackbarPos('LV','Tracking')
    # get the upper limit of hsv from trackbar
    uh = cv.getTrackbarPos('UH','Tracking')
    us = cv.getTrackbarPos('US','Tracking')
    uv = cv.getTrackbarPos('UV','Tracking')

    # Create lower and upper hsv np arrays
    lower = np.array([lh,ls,lv])
    upper = np.array([uh,us,uv])
    # create a mask for using hsv values from trackbar
    mask = cv.inRange(hsv,lower,upper)

    # Mask the original image with the mask
    res = cv.bitwise_and(frame,frame,mask=mask)

    # Show all the intermediate images and original image
    cv.imshow('Original',frame)
    cv.imshow('Mask',mask)
    cv.imshow('Result',res)

    # Exit the loop if 'q' is pressed 
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()