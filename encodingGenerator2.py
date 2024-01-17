import os

import face_recognition
import pickle
import numpy as np
import cv2

def encodingImanges(KnownImageFolderPaths):
    encodedImagesList=[]
    for images in KnownImageFolderPaths:
        images=cv2.cvtColor(images,cv2.COLOR_BGR2RGB)
        encodedImagesList.append(face_recognition.face_encodings(images)[0])

    return encodedImagesList


    pass
FolderPath='images'
imagesPath=os.listdir(FolderPath)
print(imagesPath)
KnownImageFolderPaths=[]
studentID=[]
for image in imagesPath:
    KnownImageFolderPaths.append(cv2.imread(os.path.join(FolderPath,image)))
    studentID.append(os.path.splitext(image)[0])
KnownImageEncodedList=encodingImanges(KnownImageFolderPaths)
KnownImageEncodedListwithStudentIDs=KnownImageEncodedList,studentID

with open('encodeFile2.p','wb') as file:
    pickle.dump(KnownImageEncodedListwithStudentIDs,file)



