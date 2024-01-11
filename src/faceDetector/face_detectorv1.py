from facenet_pytorch import MTCNN
from PIL import Image
import torch
from imutils.video import webcamvideostream
import cv2
import time
import glob
from tqdm.notebook import tqdm
import os
import numpy as np

from facenet_pytorch import InceptionResnetV1

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self, stride, resize=1, *args, **kwargs):

        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames,return_tlbr=True):
        """Detect faces in frames using strided MTCNN."""
        frames = [frames]
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                    for f in frames
            ]
        
                    
        boxes, probs = self.mtcnn.detect(frames[::self.stride])


        faces = []
        coordinates = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                facee = frame[box[1]:box[3], box[0]:box[2]]
                faces.append(facee)
                coordinates.append([box[1],box[0],box[3],box[2]]) # t, l, b, r
                # coordinates.append(box)
        if return_tlbr:
            return coordinates
        return faces, coordinates








def run_detection(fast_mtcnn, video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        faces, coordinates = fast_mtcnn(frame)

        for i in range(len(faces)):
            x, y, w, h = coordinates[i]
            # Blue color in BGR
            color = (0, 0, 255)
            thickness = 2
            start_point = (x, y)
            end_point = (w, h)            
            cv2.rectangle(frame, start_point, end_point, color, thickness)

        cv2.imshow('video', faces)

        if cv2.waitKey(1) &0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    fast_mtcnn = FastMTCNN(
        stride=4,
        resize=1,
        margin=14,
        factor=0.6,
        keep_all=True,
        device=device
    )
    run_detection(fast_mtcnn, 0)
