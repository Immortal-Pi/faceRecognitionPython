import cv2
import cvzone
import pickle
import os

import face_recognition
import numpy as np

if __name__=='__main__':

    cam=cv2.VideoCapture(0)
    cam.set(3,1080)
    cam.set(4,720)
    with open('encodeFile2.p','rb') as file:
        KnownImageEncodedListwithStudentIDs=pickle.load(file)
        KnownImageEncodedList,studentIDs=KnownImageEncodedListwithStudentIDs


    while True:

        success, img=cam.read()
        imgS=cv2.resize(img,(0,0),None,0.25,0.25)
        imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
        encodeFaceCurFrame=face_recognition.face_locations(imgS)
        encodeCurFrame=face_recognition.face_encodings(imgS,encodeFaceCurFrame)

        for Faceframe,faceLoc in zip(encodeCurFrame,encodeFaceCurFrame):
            matches=face_recognition.compare_faces(KnownImageEncodedList,Faceframe)
            distance=face_recognition.face_distance(KnownImageEncodedList,Faceframe)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox=x1,y1,x2-x1,y2-y1
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            # cvzone.cornerRect(img,bbox,rt=0)
            matchesIndex=np.argmin(distance)
            if matches[matchesIndex]:
                print(f'known face detected:{studentIDs[matchesIndex]}')



        cv2.imshow('camera',img)
        cv2.waitKey(1)
