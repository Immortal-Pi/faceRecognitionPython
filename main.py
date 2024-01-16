import os
import pickle

import cv2
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

        #merge the webcam on background system
        # imageBackground[0:0 + 240, 90:90 + 320] = img
        imageBackground[0:0 + 240, 90:90 + 320]=img
        # cv2.imshow("webcam",img)
        cv2.imshow("Face attendance",imageBackground)
        cv2.waitKey(1)

    # print('hello')


