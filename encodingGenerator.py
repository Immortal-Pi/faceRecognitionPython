import os

import cv2
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred=credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred,{
    'databaseURL':"https://facerecognitionrealtimepython-default-rtdb.firebaseio.com/",
    'storageBucket':"facerecognitionrealtimepython.appspot.com"
})
folderImagesPath='images'
imagesNameList=os.listdir(folderImagesPath)
imgagePath=[]
studentIDs=[]
for image in imagesNameList:
    imgagePath.append(cv2.imread(os.path.join(folderImagesPath,image)))
    # studentIDs.append(image.split('.')[0])
    studentIDs.append(os.path.splitext(image)[0])
    fileName=f'{folderImagesPath}/{image}'
    print(fileName)
    buket=storage.bucket()
    blob=buket.blob(fileName)
    blob.upload_from_filename(fileName)




def findEncodings(imagePath):

    encodeList =[]
    for image in imagePath:

        """
        change color 
        face_Recognition.face_encodings(image)
        """
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)

    return encodeList

print("encoding started")
encodeListKnown = findEncodings(imgagePath)
# print(encodeListKnown)
encodingListKnownWithIDs=[encodeListKnown,studentIDs]
# print(encodingListKnownWithIDs)
print("Encoding complete")

'''
put the encoding to the file 

'''
with open("encodeFile.p",'wb') as file:
    pickle.dump(encodingListKnownWithIDs,file)