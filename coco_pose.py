import numpy as np
import cv2
import math

# Specify the paths for the 2 files
protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel" 
# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
print("model loaded")

def get_pose(frame):
    # Read image
    #frame = cv2.imread("in.png")
    frameHeight, frameWidth, channels = frame.shape
 
    # Specify the input image dimensions
    inWidth = 368
    inHeight = 368
 
    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
 
    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)

    out = net.forward()

    H = out.shape[2]
    W = out.shape[3]

    ###################################################################################################
    #Get the point of the nose
    noseMap = out[0, 0, :, :]
    __, prob_nose, __, nose_point = cv2.minMaxLoc(noseMap)
    x_nose = (frameWidth * nose_point[0]) / W
    y_nose = (frameHeight * nose_point[1]) / H
    if prob_nose > 0.007 :
        cv2.circle(frame, (int(x_nose), int(y_nose)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

    #Get the point of the neck
    neckMap = out[0, 1, :, :]
    __, prob_neck, __, neck_point = cv2.minMaxLoc(neckMap)
    x_neck = (frameWidth * neck_point[0]) / W
    y_neck = (frameHeight * neck_point[1]) / H
    if prob_neck > 0.007 :
        cv2.circle(frame, (int(x_neck), int(y_neck)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

    #calculate the angle between nose and neck
    if x_nose == x_neck :
        face_angle = 90
    else:    
        face_angle = math.degrees(math.atan(abs(y_neck - y_nose)/abs(x_nose - x_neck)))
    #print("Face_ang",face_angle)

    ####################################################################################################

    #Get the point of the left shoulder
    lsMap = out[0, 5, :, :]
    __, prob_ls, __, ls_point = cv2.minMaxLoc(lsMap)
    x_ls = (frameWidth * ls_point[0]) / W
    y_ls = (frameHeight * ls_point[1]) / H
    if prob_ls > 0.007 :
        cv2.circle(frame, (int(x_ls), int(y_ls)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

    #Get the point of the right shoulder
    rsMap = out[0, 2, :, :]
    __, prob_rs, __, rs_point = cv2.minMaxLoc(rsMap)
    x_rs = (frameWidth * rs_point[0]) / W
    y_rs = (frameHeight * rs_point[1]) / H
    if prob_rs > 0.007 :
        cv2.circle(frame, (int(x_rs), int(y_rs)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

    #calculate the angle between nose and neck
    if y_rs == y_ls :
        shoulder_angle = 0
    else:    
        shoulder_angle = math.degrees(math.atan(abs(y_rs - y_ls)/abs(x_rs - x_ls)))
        
    #print("Shoulder_ang",shoulder_angle)

    return frame,face_angle,shoulder_angle
    

cap = cv2.VideoCapture('vid.avi')

positive_score = 0
negative_score = 0

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img,f_a,s_a = get_pose(frame)

    if f_a < 90 or s_a > 10:
        negative_score = negative_score + 1
    else:
        positive_score = positive_score + 1
    
    print(score)
    #cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


