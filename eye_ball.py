from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math

def eye_ball_size(img_gray):
        d = 10
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("C://Users//Dinusha//Desktop//fasial Emotion//real-time-facial-landmarks//shape_predictor_68_face_landmarks.dat")
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
 
                
                for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

                        #take only the right eye
                        if (name == 'right_eye'):
               
                                # extract the ROI of the face region as a separate image
                                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                                roi = gray[y-d:y + h+d, x-d:x + w+d]
                        
                                gray_blurred = cv2.blur(roi, (3, 3)) 

                                detected_circles = cv2.HoughCircles(gray_blurred,cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                                                                       param2 = 30, minRadius = 1, maxRadius = 40) 
  
                                if detected_circles is None:
                                        print("eye ball is not clear")
         
                                # Draw circles that are detected. 
                                if detected_circles is not None:
                                        # Convert the circle parameters a, b and r to integers. 
                                        detected_circles = np.uint16(np.around(detected_circles)) 
  
                                        for pt in detected_circles[0, :]:
                                                a, b, r = pt[0], pt[1], pt[2] 
                                                #cv2.circle(roi, (a, b), r, (0, 255, 0), 2)
                                                return math.pi*r*r
                                        
                                                #cv2.imshow("Detected Circle", roi) 
                                                #cv2.waitKey(0) 

cap = cv2.VideoCapture('vid.avi')

score = 0

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    image = imutils.resize(frame, width=500)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pupil_area = eye_ball_size(gray)

    if pupil_area is not None and pupil_area > 300:
        score = score + 1
    
    print(score)
    #cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()







