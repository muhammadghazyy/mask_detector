import cv2

img= cv2.imread('C:/Users/Pandu/Pictures/Makara_UI.png', -1)

cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

