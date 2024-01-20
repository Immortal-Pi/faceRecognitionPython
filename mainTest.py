import cv2
import cvzone
import pickle
import os
import face_recognition
import firebase_admin
import numpy as np
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db
if __name__=='__main__':

    camera=cv2.VideoCapture(0)
    camera.set(3,1080)
    camera.set(4,720)

    cred=credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred,{
        'databaseURL': "https://facerecognitionrealtimepython-default-rtdb.firebaseio.com/",
        'storageBucket': "facerecognitionrealtimepython.appspot.com"
    })
    bucket=storage.bucket()
    with open('encodeFile2.p','rb') as file:
        KnowImageEncodeListwithStudentID=pickle.load(file)
        KnowImageEncodeList,studentIDs=KnowImageEncodeListwithStudentID


    while True:

        success,image=camera.read()
        imageSmall = cv2.resize(image,(0,0),None,0.25,0.25)
        imageSmall = cv2.cvtColor(imageSmall,cv2.COLOR_BGR2RGB)
        CurFaceFaceLocationEncode=face_recognition.face_locations(imageSmall)
        CurFrameEncode=face_recognition.face_encodings(imageSmall,CurFaceFaceLocationEncode)

        for faceFrame,faceLocation in zip(CurFrameEncode,CurFaceFaceLocationEncode):
            match=face_recognition.compare_faces(KnowImageEncodeList,faceFrame)
            distance=face_recognition.face_distance(KnowImageEncodeList,faceFrame)
            y1,x2,y2,x1 = faceLocation
            x1,x2,y1,y2=x1*4,x2*4,y1*4,y2*4
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),thickness=2)

            MatchIndex=np.argmin(distance)

            if match[MatchIndex]:
                StudentInfo=db.reference(f'StudentsTest2/{studentIDs[MatchIndex]}').get()

                cv2.putText(image,f"NAME: {StudentInfo['name']}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,thickness=2,color=(0,0,0))
                cv2.putText(image, f'ID:{studentIDs[MatchIndex]}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2,color=(0, 0, 0))
                cv2.putText(image, f"AGE: {StudentInfo['age']}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2,
                            color=(0, 0, 0))
                cv2.putText(image, f"WEIGHT: {StudentInfo['weight']}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2,
                            color=(0, 0, 0))
                cv2.putText(image, f"HEIGHT: {StudentInfo['height']}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2,
                            color=(0, 0, 0))
                cv2.putText(image, f"GENDER: {StudentInfo['gender']}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2,
                            color=(0, 0, 0))

                blob=bucket.get_blob(f'images/{studentIDs[MatchIndex]}.jpg')
                array=np.frombuffer(blob.download_as_string(),np.uint8)
                imageFromDatabase=cv2.imdecode(array,cv2.COLOR_BGR2RGB)
                image[50:50+293,650:650+216]=imageFromDatabase

        cv2.imshow("Face", image)
        cv2.waitKey(1)



