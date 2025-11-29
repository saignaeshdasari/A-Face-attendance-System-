import cv2 
import face_recognition
import os
import pickle
import cvzone

folderPath = r'Face recoginazation\images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentid = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
     

def findEncodings(imgList):
    encodeList = []
    for img in imgList:
     img  = cv2.cvtColor(img.cv2.COLOR_8GBR2RGB)    
     encode = face_recognition.face_encodings(img)[0] 
     encodeList.append(encode)

     return encodeList     
    
encodeListknown = findEncodings(imgList)
encodeListknownwithids = [encodeListknown,studentid]

file = open("Encodedfile.p",'wb')
pickle.dump(encodeListknownwithids,file)
file.close()