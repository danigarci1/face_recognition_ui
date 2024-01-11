import cv2
import stow
import typing
import numpy as np
import onnxruntime as ort
import os 
import json
font = cv2.FONT_HERSHEY_DUPLEX
from faceDetector import face_detector



class SaveEncodings:
    def __init__(
        self, 
        detector: object,
        onnx_model_path: str = "assets/models/facenet512_weights.onnx", 
        anchors: typing.Union[str, dict] = 'faces',
        force_cpu: bool = False,
        threshold: float = 0.6,
        color: tuple = (255, 255, 255),
        thickness: int = 2,
        faces_path: str = 'assets/faces'
        ) -> None:

        if not stow.exists(onnx_model_path):
            raise Exception(f"Model doesn't exists in {onnx_model_path}")

        self.detector = detector
        self.threshold = threshold
        self.color = color
        self.thickness = thickness
        self.faces_path = faces_path

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        providers = providers if ort.get_device() == "GPU" and not force_cpu else providers[::-1]

        self.ort_sess = ort.InferenceSession(onnx_model_path, providers=providers)

        self.input_shape = self.ort_sess._inputs_meta[0].shape[1:3]

        encoding_file = 'encodings.json'
        print("[INFO]:","Loading Encodings")
        


    def normalize(self, img: np.ndarray) -> np.ndarray:
        mean, std = img.mean(), img.std()
        return (img - mean) / std

    def l2_normalize(self, x: np.ndarray, axis: int = -1, epsilon: float = 1e-10) -> np.ndarray:
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output

    def detect_save_faces(self, image: np.ndarray, output_dir: str = "faces"):
        face_crops = [image[t:b, l:r] for t, l, b, r in self.detector(image, return_tlbr=True)]

        if face_crops == []: 
            return False

        stow.mkdir(output_dir)

        for index, crop in enumerate(face_crops):
            output_path = stow.join(output_dir, f"face_{str(index)}.png")
            cv2.imwrite(output_path, crop)
            print("Crop saved to:", output_path)
        
        self.anchors = self.GenerateEncodings(output_dir)
        return True

    def load_anchors(self, faces_path: str):
        anchors = {}
        if not stow.exists(faces_path):
            return {}
        count = 0
        for folders in os.listdir(faces_path):
            for names in os.listdir(os.path.join(faces_path,folders)):
                
                anchors[folders+f"_{count}"] = self.encode(cv2.imread(os.path.join(faces_path,folders,names)))
                count += 1  

        return anchors
    

    def encode(self, face_image: np.ndarray) -> np.ndarray:
        face = self.normalize(face_image)
        face = cv2.resize(face, self.input_shape).astype(np.float32)

        encode = self.ort_sess.run(None, {self.ort_sess._inputs_meta[0].name: np.expand_dims(face, axis=0)})[0][0]
        normalized_encode = self.l2_normalize(encode)

        return normalized_encode


    
    def createEncodings(self,faces_path: str):
        anchors = {}
        if not stow.exists(faces_path):
            return {}
        count = 0
        for folders in os.listdir(faces_path):
            print(folders)
            for names in os.listdir(os.path.join(faces_path,folders)):
                frame = cv2.imread(os.path.join(faces_path,folders,names))
                face_crops = {index: {"name": "", "tlbr": tlbr} for index, tlbr in enumerate(self.detector(frame, return_tlbr=True))}
                for key, value in face_crops.items():
                    t, l, b, r = value["tlbr"]
                    anchors[folders+f"_{count}"] = self.encode(frame[t:b, l:r]).tolist()
                # anchors[folders+f"_{count}"] = self.encode(cv2.imread(os.path.join(faces_path,folders,names))).tolist()
                count += 1  

        with open('assets/encodings/encodings.json','w') as f:
            json.dump(anchors,f)



    def __call__(self):
        self.runencodings = self.createEncodings(faces_path=self.faces_path)
        
            

face_detector = face_detector.FaceDetection()



x = SaveEncodings(detector=face_detector)
x()