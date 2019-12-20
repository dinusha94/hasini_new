import argparse
import sys
import os
from utils import *
import warnings
warnings.filterwarnings("ignore")
import time
from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from keras.models import load_model


if not os.path.exists('outputs/'):
    os.makedirs('outputs/')
    
net = cv2.dnn.readNetFromDarknet('./cfg/yolov3-face.cfg', './model-weights/yolov3-wider_16000.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

root1 = tk.Tk()
root1.withdraw()

file_path = filedialog.askopenfilenames()

cap = cv2.VideoCapture(file_path[0])

print("Get redy for take photos, we'll take 5 photos in five angles")
time.sleep(4)
i =0

def crop_face(frame):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),[0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(get_outputs_names(net))
    f = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    w=f[0][2]
    h=f[0][3]
    x1 =f[0][0]
    y1 =f[0][1]
    roi = frame[y1:y1+h,x1:x1+w]
    return roi

for i in range(0,5):
    print("redy")
    time.sleep(2)
    ret, frame = cap.read()
    sample_face = crop_face(frame)  
    cv2.imwrite('outputs/out_'+str(i)+'.jpg', sample_face.astype(np.uint8))
    i = i+1
    print("next")

print("preshoot is done")


def triplet_loss(y_true, y_pred, alpha = 0.3):
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss

FRmodel = load_model('FR_net_1.h5', custom_objects={'triplet_loss': triplet_loss})
print("FR Model loaded")


database = {}
for file in glob.glob("outputs/*"):
    identity = os.path.splitext(os.path.basename(file))[0]
    database[identity] = img_path_to_encoding(file, FRmodel)

print("youre database is prepared")
    

def who_is_it(image, database, model):
    encoding = img_to_encoding(image, model)
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc - encoding)
        #print('distance for %s is %s' %(name, dist))
        return dist
    
positive_score = 0        
negative_score = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    face_img = crop_face(frame)
    distance = who_is_it(face_img, database, FRmodel)

    if distance > 0.7:
       negative_score = negative_score + 1

    elif distance < 0.7:
        positive_score = positive_score + 1
        
    #print(positive_score,negative_score)
    cv2.putText(face_img, str(distance), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0) )    
    cv2.imshow('frame',face_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

root = tk.Tk()
root.geometry("300x70")
root.title("Emotional analysis")    
lbl1=Label(root, text="Positive score :"+ str(positive_score))
lbl1.pack()
lbl2=Label(root, text="Negative score :"+ str(negative_score))
lbl2.pack()
root.mainloop()    
