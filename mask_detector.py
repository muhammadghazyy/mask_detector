import cv2
import numpy as np

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

capture = cv2.VideoCapture(0) # 0 is acccesing the first webcam
 
while True:
    ret, frame = capture.read()
    #frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    #frame = cv2.resize(frame , (765,500))
    width = int(capture.get(3))
    height = int(capture.get(4))

    img = cv2.line(frame, (15,15) , (width, height), (0,255,0), 15)
    img = cv2.line(img, (0,height) , (width, 0), (255,255,0), 15)
    img = cv2.rectangle(img, (100,100), (150,150),(128,128,128),2)
    img = cv2.circle(img, (width//2, height//2) , 75, (255,200,150), 3)
    
    font = cv2.FONT_HERSHEY_TRIPLEX
    img = cv2.putText(img, 'MUHAMMAD GHAZY 1806193445', (1,height//2), font, 5, (0,0,0), 2, cv2.LINE_AA)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
