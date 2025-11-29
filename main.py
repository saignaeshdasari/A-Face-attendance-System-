import cv2 
import pickle
import numpy as np 
import os 
import cvzone
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

imgBackground = cv2.imread(r"Face recoginazation\resources\background.png")

folderPath = r"Face recoginazation\resources\Modes"
modepathList = os.listdir(folderPath)
imgModeList = []
for path in modepathList:
    imgModeList.append(cv2.imread(os.path.join(folderPath,path)))


file = open('Encoderfile.p','rb')
encodeListknownwithids = pickle.load(file)
file.close()
encodeListknown,studentid = encodeListknownwithids


while True:
    success, img = cap.read()

    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS  = cv2.cvtColor(img.cv2.COLOR_8GBR2RGB)   

    faceCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS,faceCurrFrame) 

    imgBackground[162:162+480,55:55+640] = img
    imgBackground[44:44+633,808:808+414] = imgModeList[2]
    for encodeFace, faceloc in zip(encodeCurrFrame,faceCurrFrame):
        matches = face_ecognition.compare_faces(encodeListknown,encodeFace)
        facesDis = face_ecognition.face_distance(encodeListknown,encodeFace)
        print(matches,"matches")
        print(facesDis,"faceDis")

        matchIndex = np.argmin(facesDis)

        if matches[matchIndex]:
            print("Known face detected")
            print(studentid[matchIndex])
        y1,x2,y2,x1 = faceloc
        y1,x2,y2,x1 = y1+4,x2+4,y2+4,x1+4 
        bbox = 55+x1,162+y1, x2-x1,y2-y1 
        imgBackground =cvzone.cornerRect(imgBackground,bbox,rt=0)    

    # cv2.imshow("Webcam",img)
    cv2.imshow("Face attendance",imgBackground)
    if cv2.waitKey(10) & 0xFF == ord('a'):
        break
