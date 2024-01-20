import os

import face_recognition
import pickle

import firebase_admin.credentials
import numpy as np
import cv2
from firebase_admin import storage
def encodeImagesandPath(imagePathNames):
    encode=[]
    for image in imagePathNames:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        encode.append(face_recognition.face_encodings(image)[0])

    return encode

cread=firebase_admin.credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cread,{
    'databaseURL':"https://facerecognitionrealtimepython-default-rtdb.firebaseio.com/",
    "storageBucket":"facerecognitionrealtimepython.appspot.com"
})

imageFolderPath='images'
imagePathNames=[]
imageNames=os.listdir(imageFolderPath)
studentID=[]
for image in imageNames:
    imagePathNames.append(cv2.imread(os.path.join(imageFolderPath,image)))
    studentID.append(os.path.splitext(image)[0])
    bucket=storage.bucket()
    fileName=f'{imageFolderPath}/{image}'
    blob=bucket.blob(fileName)
    blob.upload_from_filename(fileName)
KnowImageEncodeList=encodeImagesandPath(imagePathNames)
KnowImageEncodeListwithStudentID=KnowImageEncodeList,studentID

with open('encodeFile2.p','wb') as file:
    pickle.dump(KnowImageEncodeListwithStudentID,file)
# print(KnowImageEncodeList)