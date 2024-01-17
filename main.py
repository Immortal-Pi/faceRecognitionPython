import os
import pickle
import cv2
import face_recognition
import numpy as np
import cvzone

if __name__ == '__main__':
    """
    cv2.videoCapture - points to the camera from which video will be taken
    cv2 object.set - sets the image dimensions 
    cap.read - reads the camera
    
    set image resolution
    cap.set(3,4) - its the pos
    """
    cap=cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4,240)

    imageBackground=cv2.imread("Resources/background.png")

    #folder for modes after successful face recognition
    folderModePath='Resources/modes'
    modePathList = os.listdir(folderModePath)

    imgModeList =[]
    for path in modePathList:
        imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

    #load the encoding file
    with open('encodeFile.p','rb') as file:
        encodeListKnowWithIDs=pickle.load(file)
        encodeListKnow,studentIDs=encodeListKnowWithIDs
        # print(studentIDs)
    print("encodeFileLoaded")

    while True:
        """
        cap.read() - captures frames from the webcam
        success - True if the frame is captured 
        img - frame captured 
        imshow - displays the captured frame i.e. img
        waitkey(1) - delay time
        
        """
        success, img=cap.read()
        #resize the image(scale the image to a smaller size) 0.25 is the scale value
        imgS=cv2.resize(img,(0,0), None,0.25,0.25)
        imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

        faceCurrentFrame=face_recognition.face_locations(imgS) #used to detect face location in the frame imgS
        encodeCurFrame=face_recognition.face_encodings(imgS,faceCurrentFrame)

        #merge the webcam on background system
        # imageBackground[0:0 + 240, 90:90 + 320] = img
        imageBackground[0:0 + 240, 90:90 + 320]=img

        for encodeFace, faceLoc in zip(encodeCurFrame,faceCurrentFrame):
            matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
            print(f"matches :{matches} \n facedistance: {faceDis}")
            matchIndex=np.argmin(faceDis)
            # print(f"match Index {matchIndex}")
            if matches[matchIndex]:
                print(f"known face detected \n {studentIDs[matchIndex]}")
                y1,x2,y2,x1=faceLoc
                y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
                bbox = 90+x1, y1, x2-x1,y2-y1
                cvzone.cornerRect(imageBackground,bbox,rt=0) #mark face


        # cv2.imshow("webcam",img)
        cv2.imshow("Face attendance",imageBackground)
        cv2.waitKey(1)

    # print('hello')


