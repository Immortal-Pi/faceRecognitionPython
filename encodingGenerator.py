import os
import pickle

import face_recognition
import firebase_admin
import cv2
from firebase_admin import storage
from firebase_admin import db

def encodeGenerator(imageNameList):
    """

    :param imageNameList: List of image readable format
    :return: encoded list

    """
    encode=[]
    for image in imageNameList:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        encode.append(face_recognition.face_encodings(image)[0])

    return encode

imageFolderPath = 'images'
imageNames = os.listdir(imageFolderPath)
imageNamePath=[]
studentIDs=[]

cred=firebase_admin.credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://facerecognitionrealtimepython-default-rtdb.firebaseio.com/",
    "storageBucket":"facerecognitionrealtimepython.appspot.com"
})
bucket = storage.bucket()

for image in imageNames:
    imageNamePath.append(cv2.imread(os.path.join(imageFolderPath,image)))
    studentIDs.append(os.path.splitext(image)[0])
    fileName=f"{imageFolderPath}/{image}"
    blob=bucket.blob(fileName)
    blob.upload_from_filename(fileName)

KnowImageEncodeList = encodeGenerator(imageNamePath)
KnowImageEncodeWithStudentIDs = KnowImageEncodeList,studentIDs

with open('encodeFile.p','wb') as file:
    pickle.dump(KnowImageEncodeWithStudentIDs,file)