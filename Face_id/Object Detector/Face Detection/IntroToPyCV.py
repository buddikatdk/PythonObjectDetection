import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0);

while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.1,5);
    for(x,y,p,q) in faces:
      cv2.rectangle(img,(x,y),(x+p,y+q),(0,255,0),2)
        
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
            break;
cam.release()
cv2.destroyAllWindows()

