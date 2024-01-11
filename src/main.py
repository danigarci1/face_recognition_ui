
#Data manipulation includes
import cv2
import numpy as np
import time
#ML includes
from ultralytics import YOLO

#Project includes
from faceDetector import face_detector
from faceRecognition import face_recognition
import torch
from multiprocessing import Queue
from multiprocessing import Process, Manager
from utils.log import loggers
from utils.draw import drawPerson
import configparser
import argparse
np.random.seed(5)
font = cv2.FONT_HERSHEY_DUPLEX

def run_image(image_path,model_path,recognition_model_path,detection_confidence,recognition_confidence):
    frame = cv2.imread(image_path)
    detector = face_detector.FaceDetection()
    recog = face_recognition.FaceNet(
        detector=detector,
        threshold=recognition_confidence,
        onnx_model_path = recognition_model_path)
    yolov8_detector = YOLO(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolov8_detector.to(device)
    # Capture frame-by-frame
    # if the input queue *is* empty, give the current frame to
    # classify

    result = yolov8_detector.track(frame,verbose=False,persist=True,conf=detection_confidence)[0] #Verbose False to avoid yolov8 messages

    for output in result.cpu().numpy().boxes.data:
        bbox_left = int(output[0])
        bbox_top = int(output[1])
        bbox_w = int(output[2]) 
        bbox_h = int(output[3])
        if output.shape[0] == 7:
            id = int(output[4])
            prev_id = id
        else:
            id =prev_id
        name = "undefined"
        if bbox_w > 0 and bbox_h > 0:
            person_frame = frame[bbox_top:bbox_h,bbox_left:bbox_w,:]
            name = recog(frame=person_frame,face_frame=True)

        drawPerson(frame,bbox_left,bbox_top,bbox_w,bbox_h,name,(0,255,255))  


    # out.write(frame) #Comment this line if you dont want to save result
    cv2.imshow('frame', frame)
    cv2.waitKey(-1)

    cv2.destroyAllWindows()


INPUT_SIZE = 640

def run_video(video_file_path,model_path,recognition_model_path,detection_confidence,recognition_confidence):
    manager = Manager()
    id_face_dictionary = manager.dict()
    vid = cv2.VideoCapture(video_file_path)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    frameWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_ratio = frameWidth / INPUT_SIZE
    h_ratio = frameHeight / INPUT_SIZE
    loggers["general"].info(f"[info] W, H, FPS\n{frameWidth}, {frameHeight}, {vid.get(cv2.CAP_PROP_FPS)}")

    inputQueue = Queue(maxsize=1)
    outputQueue = Queue(maxsize=1)
    recognitionQueue = Queue()
    p = Process(target=object_detection_, args=(model_path,detection_confidence,inputQueue, outputQueue,recognitionQueue,))
    p.daemon = True
    p.start()

    pRec = Process(target=recognize_algorithm, args=(recognition_model_path,recognitionQueue,id_face_dictionary,recognition_confidence,))
    pRec.daemon = True
    pRec.start()

    time.sleep(5)
    # time the frame rate....
    timer1 = time.time()
    frames = 0
    queuepulls = 0
    timer2 = 0
    t2secs = 0
    fps = 0.0
    qfps = 0.0
    queuepulls = 0.0
    out = None
    id = 0
    prev_id = 0
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            if queuepulls == 1:
                timer2 = time.time()
            # Capture frame-by-frame
            # if the input queue *is* empty, give the current frame to
            # classify
            if inputQueue.empty():
                yoloframe = cv2.resize(frame,(int(frameWidth/w_ratio),int(frameHeight/h_ratio)))
                inputQueue.put(yoloframe)
            else:
                loggers["general"].debug("Skipping frame from face detection")

                 
            # if the output queue *is not* empty, grab the detections
            if not outputQueue.empty():
                out = outputQueue.get()
            # else:
            #     out = None
            if out is not None:
                queuepulls += 1
                for output in out:
                    bbox_left = int(output[0]* w_ratio)
                    bbox_top = int(output[1]* h_ratio)
                    bbox_w = int(output[2]* w_ratio) 
                    bbox_h = int(output[3]* h_ratio)
                    if output.shape[0] == 7:
                        id = int(output[4])
                        prev_id = id
                    else:
                        id =prev_id
                    if id in id_face_dictionary:
                        name = id_face_dictionary[id] + " "+ str(id)
                    else:
                        name = "undefined "+str(id)
                    drawPerson(frame,bbox_left,bbox_top,bbox_w,bbox_h,name,(0,255,255))  
            # Display the resulting frame
            cv2.rectangle(frame, (0, 0),
                        (frameWidth, 20), (0, 0, 0), -1)

            cv2.rectangle(frame, (0, frameHeight-20),
                        (frameWidth, frameHeight), (0, 0, 0), -1)
            cv2.putText(frame, 'VID FPS: '+str(fps), (frameWidth-80, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, 'DET FPS: '+str(qfps), (frameWidth-80, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

            # out.write(frame) #Comment this line if you dont want to save result
            cv2.imshow('frame', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            
            # FPS calculation
            frames += 1
            if frames >= 1:
                end1 = time.time()
                t1secs = end1-timer1
                fps = round(frames/t1secs, 2)
            if queuepulls > 1:
                end2 = time.time()
                t2secs = end2-timer2
                qfps = round(queuepulls/t2secs, 2)
    # out.release()
    vid.release()
    cv2.destroyAllWindows()
    p.kill()
    pRec.kill()

def recognize_algorithm(model_path,recognitionQueue,id_face_dictionary,confidence):
    detector = face_detector.FaceDetection()
    recog = face_recognition.FaceNet(
        detector=detector,
        threshold=confidence,
        onnx_model_path = model_path)
    loggers['recognition'].info("Recognition initialized")
    while True:
        if not recognitionQueue.empty():
            out = recognitionQueue.get()
            frame = out[1]
            boxes = out[0]
            for output in boxes: 
                bbox_left = int(output[0])
                bbox_top = int(output[1])
                bbox_w = int(output[2]) 
                bbox_h = int(output[3])
                id = int(output[4])
                if bbox_w > 0 and bbox_h > 0:
                    person_frame = frame[bbox_top:bbox_h,bbox_left:bbox_w,:]
                    start_time = time.time()

                    name = recog(frame=person_frame,face_frame=True)
                    loggers['recognition'].debug(f"RECOGNITION - Inference time: {round(time.time()-start_time,2)}")

                    if bool(name):
                        to_remove = []
                        for key, value in id_face_dictionary.items():
                            if value == name:
                                if id != key:
                                        to_remove.append(key)
                                loggers["recognition"].info(f"{name} already in dict. ID: {id}")
                        for k in to_remove:
                            id_face_dictionary.pop(k)

                        #once deleted, we add new key
                        id_face_dictionary[id] = name
                        loggers["recognition"].info(f"Added {name} to key {id}")



def object_detection_(model_path,confidence,inputQueue,outputQueue,recognitionQueue):
    global id_face_dictionary
    yolov8_detector = YOLO(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolov8_detector.to(device)
    loggers['tracking'].info("Detection initialized")
    while True:
        if not inputQueue.empty():
            frame = inputQueue.get()
            result = yolov8_detector.track(frame,verbose=False,persist=True,conf=confidence)[0] #Verbose False to avoid yolov8 messages
            outputQueue.put(result.cpu().numpy().boxes.data)
            if recognitionQueue.empty():
                recognitionQueue.put((result.cpu().numpy().boxes.data,frame))
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='path to cfg file', default="config.cfg")
    config = configparser.ConfigParser()

    # Load the configuration file
    args = parser.parse_args()
    config.read(args.cfg)
    modelPath = config["detector"]["model_path"]
    camera_idx = config["video"]["camera_idx"]
    confThreshold = config["detector"].getfloat("conf_threshold")
    show = config["general"].getboolean("show")
    recognitionModelPath = config["recognition"]["model_path"]
    recognitionThreshold = config["recognition"].getfloat("recognition_threshold")
    from_video = config['detector'].getboolean('from_video')
    image_path = config['image']['image_path']
    loggers["general"].debug(f"Initializing system. Running from {camera_idx}")
    if from_video:
        run_video(video_file_path=camera_idx,
                model_path=modelPath,
                recognition_model_path=recognitionModelPath,
                detection_confidence=confThreshold,
                recognition_confidence=recognitionThreshold)
    else:
        run_image(image_path=image_path,
                model_path=modelPath,
                recognition_model_path=recognitionModelPath,
                detection_confidence=confThreshold,
                recognition_confidence=recognitionThreshold)