import cv2
import numpy as np
import os
import tensorflow as tf

# Mask Detection

input_shape =  (128,128,3)
labels_dict = {0:'Mask On' , 1:'Mask Off'}
color_dict = {0:(61,235,52) , 1:(0,0,255)}
model = tf.keras.models.load_model('mask_mnv2.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
size = 4

capture = cv2.VideoCapture(1)

while True:
    ret,frame = capture.read()
    font = cv2.FONT_HERSHEY_SIMPLEX

    #small = cv2.resize(frame ,(0,0),fx=0.5,fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,1.275,4)

    for (x,y,w,h) in face:
        face_image = frame[y:y+h, x:x+w]

        resized = cv2.resize(face_image, (input_shape[0], input_shape[1]))
        reshaped = np.reshape(resized, (1,input_shape[0], input_shape[1],3))

        result = model.predict(reshaped)

        label = np.argmax(result,axis=1)[0]

        cv2.rectangle(frame , (x,y), (x+w,y+h), color_dict[label],2)
        cv2.rectangle(frame,(x,y-40), (x+w,y), color_dict[label],-1)
        cv2.putText(frame,labels_dict[label],(x,y-10),font,1,(255,255,255),2)

    cv2.imshow("Frame",frame)

    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()