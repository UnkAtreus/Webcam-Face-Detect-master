import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

def create_dataset(img,id,img_id):
    cv2.imwrite("data/pic."+str(id)+"."+str(img_id)+".jpg",img)

def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,clf):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        gray,
        scaleFactor,
        minNeighbors
    )
    coords=[]
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        id,con=clf.predict(gray[y:y+h,x:x+w])
        if con <= 100 :
            cv2.putText(img,"Nom",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        else :
            cv2.putText(img,"Unknow",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        if (con < 100):
            con  = "  {0}%".format(round(100 - con))
        else:
            con  = "  {0}%".format(round(100 - con))
            
        
        print(str(con))
        #if id == 1:
        #    cv2.putText(img,"Nom",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        coords = [x,y,w,h]
    return img,coords

def detect(img,faceCascade,img_id,clf):
    img,coords = draw_boundary(img,faceCascade,1.1,5,(0,0,255),clf)
    
    if len(coords) == 4:
        id=1
        result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        #create_dataset(result,id,img_id)
    return img

video_capture = cv2.VideoCapture(0)
anterior = 0

img_id = 0
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = detect(frame,faceCascade,img_id,clf)
    img_id += 1

    if anterior != len(frame):
        anterior = len(frame)
        log.info("faces: "+str(len(frame))+" at "+str(dt.datetime.now()))


  
    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
