from time import time
import cv2
from faceDetector import face_detector

import os

username = input("Enter Name:")
directory = username
  
# Parent Directory path
parent_dir = os.getcwd() + '\\faces'
print(parent_dir)
path = os.path.join(parent_dir, directory)
print(path)
#mode = 0o666
try: 
    os.mkdir(path) 
except OSError as error: 
    print(error)

face_detector = face_detector.FaceDetection()


def startCapture():
    cam = cv2.VideoCapture(1)

    cv2.namedWindow("Face Capture")

    img_counter = 0
    width = 400
    height = 400
    dim = (width, height)
    while True:
        ret, frame = cam.read()
        if not ret:
            
            print("failed to grab frame")
            break
        new_frame,top,left,top_height,left_width = face_detector(frame=frame)
        if top is not None:
            cropped_image = new_frame[int(top):int(top_height),int(left):int(left_width),:]
            resized_image = cv2.resize(cropped_image,dim)
        else:
            new_frame = frame
        

        cv2.imshow("test", new_frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = path+"/_"+ str(img_counter)+ ".png"
            cv2.imwrite(img_name, resized_image)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

cv2.destroyAllWindows()


startCapture()