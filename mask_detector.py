import cv2
import numpy as np
import os
import tensorflow as tf

# img= cv2.imread('C:/Users/Pandu/Pictures/Makara_UI.png', -1)
# imgr = cv2.resize(img,(0,0),fx=0.25, fy=0.5) #shrink by 2
# imgrot = cv2.rotate(img , cv2.cv2.ROTATE_90_CLOCKWISE)

# cv2.imwrite('C:/Users/Pandu/Pictures/Makara_UI2.png',imgrot)
# cv2.imshow('Image',imgrot)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#SESSION 2
 # img itu np.array ternyata awokaowkaowk
#print(img.shape)

#Session 3 capturing camera
# capture = cv2.VideoCapture(0) # 0 is acccesing the first webcam
 
# while True:
#     ret, frame = capture.read()
#     #frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
#     #frame = cv2.resize(frame , (765,500))
#     width = int(capture.get(3))
#     height = int(capture.get(4))

#     small_frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
#     canvas = np.zeros(frame.shape, np.uint8)
#     canvas[:height//2 , :width//2]  = cv2.rotate(small_frame, cv2.cv2.ROTATE_180)
#     canvas[height//2: , width//2:]  = small_frame
#     canvas[:height//2 , width//2:]  = cv2.rotate(small_frame, cv2.cv2.ROTATE_180)
#     canvas[height//2: , :width//2]  = small_frame

#     cv2.imshow('frame', canvas)

#     if cv2.waitKey(1) == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows()


# SESSION 4

# capture = cv2.VideoCapture(0) # 0 is acccesing the first webcam
 
# while True:
#     ret, frame = capture.read()
#     #frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
#     #frame = cv2.resize(frame , (765,500))
#     width = int(capture.get(3))
#     height = int(capture.get(4))

#     img = cv2.line(frame, (15,15) , (width, height), (0,255,0), 15)
#     img = cv2.line(img, (0,height) , (width, 0), (255,255,0), 15)
#     img = cv2.rectangle(img, (100,100), (150,150),(128,128,128),2)
#     img = cv2.circle(img, (width//2, height//2) , 75, (255,200,150), 3)
    
#     font = cv2.FONT_HERSHEY_TRIPLEX
#     img = cv2.putText(img, 'MUHAMMAD GHAZY 1806193445', (1,height//2), font, 5, (0,0,0), 2, cv2.LINE_AA)

#     cv2.imshow('frame', img)

#     if cv2.waitKey(1) == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows()

# Session 5 Face detection using HAAR CASCADE
# capture = cv2.VideoCapture(0) # 0 is acccesing the first webcam
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# while True:
#     ret, frame = capture.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscalling first
#     faces = face_cascade.detectMultiScale(gray,1.3,3)
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame , (x,y), (x+w,y+h), (255,255,255), 4)
#         cv2.rectangle(frame , (x-1,y+h), (x+300, y+h+25+10),(255,255,255),2)
#         cv2.putText(frame, 'Muhammad Ghazy', (x,y+h+25), font, 1, (255,255,255), 2, cv2.LINE_AA)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray, 1.3 , 5, 3)
#         #for (ex,ey,ew,eh) in eyes:
#             #cv2.rectangle(roi_color , (ex,ey), (ex+ew , ey+eh), (255,0,0),4)

#     cv2.imshow('frame', frame)

#     if cv2.waitKey(1) == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows()

# Mask Detection

input_shape =  (128,128,3)
labels_dict = {0:'Mask On' , 1:'Mask Off'}
color_dict = {0:(61,235,52) , 1:(0,0,255)}
model = tf.keras.models.load_model('mask_mnv2.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
size = 4

capture = cv2.VideoCapture(0)

while True:
    ret,frame = capture.read()
    font = cv2.FONT_HERSHEY_SIMPLEX

    #small = cv2.resize(frame ,(0,0),fx=0.5,fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,1.3,3)

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