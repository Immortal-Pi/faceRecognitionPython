import os

import cv2
import face_recognition
import pickle

folderImagesPath='images'
imagesNameList=os.listdir(folderImagesPath)
imgagePath=[]
studentIDs=[]
for image in imagesNameList:
    imgagePath.append(cv2.imread(os.path.join(folderImagesPath,image)))
    # studentIDs.append(image.split('.')[0])
    studentIDs.append(os.path.splitext(image)[0])

print(studentIDs)


def findEncodings(imagePath):

    encodeList =[]
    for image in imagePath:

        """
        change color 
        
        """
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)

    return encodeList

print("encoding started")
encodeListKnown = findEncodings(imgagePath)
# print(encodeListKnown)
encodingListKnownWithIDs=[encodeListKnown,studentIDs]
# print(encodingListKnownWithIDs)
print("Encoding complete")

with open("encodeFile.p",'wb') as file:
    pickle.dump(encodingListKnownWithIDs,file)