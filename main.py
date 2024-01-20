import os
import pickle
import numpy as np
import face_recognition
import firebase_admin
import cv2
from firebase_admin import storage
from firebase_admin import db



if __name__=='__main__':
    '''
    get camera attributes and set the resolution
    '''
    camera=cv2.VideoCapture(0)
    camera.set(3,1080)
    camera.set(4,720)

    with open("encodeFile.p",'rb') as file:
        KnowImageEncodeWithStudentIDs=pickle.load(file)
    KnowImageEncodeList,studentIDs = KnowImageEncodeWithStudentIDs

    cred = firebase_admin.credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://facerecognitionrealtimepython-default-rtdb.firebaseio.com/",
        'storageBucket': "facerecognitionrealtimepython.appspot.com"
    })
    bucket = storage.bucket()


    while True:

        '''
        get the video input using .read()
        reduce the scale of the image to make it easier for processing
                
        '''
        success, image = camera.read()
        imageSmall = cv2.resize(image,(0,0),None,0.25,0.25)
        imageSmall = cv2.cvtColor(imageSmall,cv2.COLOR_BGR2RGB)

        '''
        get the face location and encode the frame with face_recognition library
        '''
        curFaceFrame = face_recognition.face_locations(imageSmall)
        curFaceFrameEncoding = face_recognition.face_encodings(imageSmall,curFaceFrame)

        for faceFrame,faceLoc in zip(curFaceFrameEncoding,curFaceFrame):
            '''
                match the Knowface with camera face
                get the distance i.e. the degree of matching 
            '''
            matches=face_recognition.compare_faces(KnowImageEncodeList,faceFrame)
            distance=face_recognition.face_distance(KnowImageEncodeList,faceFrame)

            '''
                we can get the coordinates for identify the face 
                cv2.rectangle for making green lines around face
            '''
            y1,x2,y2,x1 = faceLoc
            x1,y1,x2,y2 = x1*4,y1*4,x2*4,y2*4
            cv2.rectangle(image,(x1,y1),(x2,y2),color=(0,255,0),thickness=2)

            MatchIndex=np.argmin(distance)
            if matches[MatchIndex]:

                StudentInfo=db.reference(f"Students/{studentIDs[MatchIndex]}").get()
                cv2.putText(image, f"NAME: {StudentInfo['name']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2,
                            color=(0, 0, 0))
                cv2.putText(image, f'ID:{studentIDs[MatchIndex]}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2,
                            color=(0, 0, 0))
                cv2.putText(image, f"MAJOR: {StudentInfo['major']}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2,
                            color=(0, 0, 0))
                cv2.putText(image, f"STANDING: {StudentInfo['standing']}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            thickness=2,
                            color=(0, 0, 0))
                cv2.putText(image, f"STARTING_YEAR: {StudentInfo['starting_year']}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            thickness=2,
                            color=(0, 0, 0))
                # cv2.putText(image, f"LAST_ATTENDANCE_TIME: {StudentInfo['last_attendance_time']}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             thickness=2,
                #             color=(0, 0, 0))
                blob=bucket.get_blob(f'images/{studentIDs[MatchIndex]}.jpg')
                array=np.frombuffer(blob.download_as_string(),np.uint8)
                imageFromDatabase=cv2.imdecode(array,cv2.COLOR_BGR2RGB)
                image[50:50 + 293, 650:650 + 216] = imageFromDatabase





        cv2.imshow('Face', image)
        cv2.waitKey(1)
