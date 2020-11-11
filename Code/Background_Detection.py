import cv2
import imutils
import numpy as np
from pymouse import PyMouse
from sklearn.metrics import pairwise
# import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
# global variables
bg = None
#-------------------------------------------------------------------------------
# Function - To find the running average over the background
#-------------------------------------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#-------------------------------------------------------------------------------
# Function - To segment the region of hand in the image
#-------------------------------------------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)
    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    cnts, ret = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return None
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#-------------------------------------------------------------------------------
# Function - To count the number of fingers in the segmented hand region
#-------------------------------------------------------------------------------
def count(thresholded, segmented):
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = (extreme_left[0] + extreme_right[0]) // 2
    cY = (extreme_top[1] + extreme_bottom[1]) // 2

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # calculate the radius of the circle with 70% of the max euclidean distance obtained
    radius = int(0.7 * maximum_distance)
    
    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # take out the circular region of interest which has 
    # the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
    
    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    cnts, rettt = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initalize the finger count
    count = 0

    # loop through the contours found
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
 
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
 
    # return the eye aspect ratio
    return ear

def eyeBrow_threshold(eyeB1,eyeB2,eye1,eye2):
    avg=0
    for i in range(4):
        avg+=dist.euclidean(eyeB1[i],eye1[i])
    for i in range(4):
        avg+= dist.euclidean(eyeB2[i],eye2[i])
        
    avg/=8
    
    return float(avg)

#Threshold for Eye Aspect Ratio(Detecting a closed eye)
earThreshold=0.15

# Threshold for a valid eyeBrow raise
BrowThreshold=5
    
#Threshold for blink duration(For ajusting left click senstivity)
clickThresholdL=3

#Threshold for eyebrow raises(For adjusting right click senstivity)
clickThresholdR=4
    
# initialize accumulated weight
accumWeight = 0.5

#mouse Speed
speed=20
    
#move threshold
moveThreshold=3


#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    #PyMouse object
    m=PyMouse()
    
    # For wink detection(pre trained dlib model)
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")  # use link to get file https://drive.google.com/open?id=1r84y_rYPyGuGiuJhXTjtPQHCpCj6w_Ct
    
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    # (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # #indxes of facial landmarks for left and right eyebrows
    # (lBStart, lBEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    # (rBStart, rBEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    
    
    
    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates(for detecting hands in that region)
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False

    #count for closed eyes in frames to detect a valid blink
    bCount=0
    
    #count for closed eyes in frames to detect a valid eyeBrow Raise
    rCount=0
    
    #Initial eye2eyeBrow distance
    initialR=0
    
    #initial Eye Aspect Ratio
    initialL=0
    
    #Initial eye2eyeBrow and aspect ratio calibration frames counter
    initCounter=0
    
    
    #Move Threshold count
    moveThresholdCountL=0
    moveThresholdCountR=0
    moveThresholdCountU=0
    moveThresholdCountD=0
    maxi=0
    
    
    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames <= 300:
            run_avg(gray, accumWeight)
               
            if num_frames == 1:
                print ("[STATUS] please wait! calibrating...")
                initial=0
                initCounter=0
                
            if num_frames == 300:
                print ("[STATUS] calibration successfull...")
                
            
            
        else:
            # segment the hand region
            hand = segment(gray)
            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                # count the number of fingers
                fingers = count(thresholded, segmented)
                print(fingers)
                if fingers==1: # & moveThresholdCountR>moveThreshold:         #right
                    moveThresholdCountR+=1
                    moveThresholdCountL=0
                    moveThresholdCountU=0
                    moveThresholdCountD=0
                    
                    # m.move(m.position()[0]+speed,m.position()[1])
                    cv2.putText(clone, "Going->Right"+ str(fingers), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                
                elif fingers==2: # & moveThresholdCountL>moveThreshold:   #left
                    moveThresholdCountL+=1
                    moveThresholdCountR=0
                    moveThresholdCountU=0
                    moveThresholdCountD=0

                    # m.move(m.position()[0]-speed,m.position()[1])
                    cv2.putText(clone, "Going->Left"+ str(fingers), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                   
                elif fingers==3: # & moveThresholdCountU>moveThreshold:   #up
                    moveThresholdCountU+=1
                    moveThresholdCountL=0
                    moveThresholdCountR=0
                    moveThresholdCountD=0
                    
                    # m.move(m.position()[0],m.position()[1]-speed)
                    cv2.putText(clone, "Going->Up"+str(fingers), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                
                elif fingers==4: # & moveThresholdCountD>moveThreshold:   #down
                    moveThresholdCountD+=1
                    moveThresholdCountL=0
                    moveThresholdCountU=0
                    moveThresholdCountR=0
                    
                    # m.move(m.position()[0],m.position()[1]+speed)
                    cv2.putText(clone, "Going->Down"+str(fingers), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                else:
                    cv2.putText(clone, "Going->Nowhere"+ str(fingers), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # show the thresholded image
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
            
        #reset background and eyeBrow Threshold
        if keypress == ord("r"):
            num_frames=0



# free up memory
camera.release()
cv2.destroyAllWindows()