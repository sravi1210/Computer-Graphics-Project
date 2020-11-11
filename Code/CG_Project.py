import cv2
import numpy as np
import dlib
import math
from pymouse import PyMouse
from math import hypot
blink = False
speed = 4
m = PyMouse()

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
	return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def eye_blink(frame):
    
    global blink
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            if not blink:
                blink = True
                return True
        else:
            blink = False

    return False

while(1):

    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    kernel = np.ones((3,3),np.uint8)
    
    #define region of interest
    roi=frame[100:400, 400:700]
    
    
    cv2.rectangle(frame,(400,100),(700,400),(0,255,0),0)    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    
     
# define range of skin color in HSV
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)
    
 #extract skin colur imagw  
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    

    
#extrapolate the hand to fill dark spots within
    mask = cv2.dilate(mask,kernel,iterations = 4)
    
#blur the image
    mask = cv2.GaussianBlur(mask,(5,5),100) 
    l=0
    try:
   	    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   	    cnt = max(contours, key = lambda x: cv2.contourArea(x))
   	    epsilon = 0.0005*cv2.arcLength(cnt,True)
   	    approx= cv2.approxPolyDP(cnt,epsilon,True)
   	    hull = cv2.convexHull(cnt)
   	    areahull = cv2.contourArea(hull)
   	    areacnt = cv2.contourArea(cnt)
   	    arearatio=((areahull-areacnt)/areacnt)*100
   	    hull = cv2.convexHull(approx, returnPoints=False)
   	    defects = cv2.convexityDefects(approx, hull)
   	    for i in range(defects.shape[0]):
   	    	s,e,f,d = defects[i,0]
   	    	start = tuple(approx[s][0])
   	    	end = tuple(approx[e][0])
   	    	far = tuple(approx[f][0])
   	    	pt= (100,180)
   	    	a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
   	    	b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
   	    	c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
   	    	s = (a+b+c)/2
   	    	ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
   	    	d=(2*ar)/a
   	    	angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
   	    	if angle <= 90 and d>30:
   	    		l += 1
   	    		cv2.circle(roi, far, 3, [255,0,0], -1)

	        cv2.line(roi,start, end, [0,255,0], 2)
	        
    except:
    	areacnt = 0
    	pass

    l+=1
    
    #print corresponding gestures which are in their ranges
    font = cv2.FONT_HERSHEY_SIMPLEX
    if l==1:
        if areacnt<2000:
            cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            value = eye_blink(frame)
            if value:
            	m.click(m.position()[0],m.position()[1], 2)
            	print("Blinked")

        else:
            cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            m.move(m.position()[0]+speed,m.position()[1]) #right

    elif l==2:
        cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        m.move(m.position()[0]-speed,m.position()[1]) #left
        
    elif l==3:
        cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        m.move(m.position()[0],m.position()[1]-speed) #up
        
    elif l==4:
        cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        m.move(m.position()[0],m.position()[1]+speed) #down
        
    elif l==5:
        cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        m.click(m.position()[0],m.position()[1], 1)

    #show the windows
    cv2.imshow('mask',mask)
    cv2.imshow('frame',frame)
        
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()