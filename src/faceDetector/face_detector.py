import cv2
import typing
import numpy as np
import mediapipe as mp

class FaceDetection:
    """Object to create and do mediapipe face detection, more about it:
    https://google.github.io/mediapipe/solutions/face_detection.html
    """
    def __init__(
        self,
        model_selection: bool = 1,
        confidence: float = 0.5,
        mp_drawing_utils: bool = True,
        color: typing.Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        ) -> None:
  
        self.mp_drawing_utils = mp_drawing_utils
        self.color = color
        self.thickness = thickness
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=model_selection, min_detection_confidence=confidence)

    def tlbr(self, frame: np.ndarray, mp_detections: typing.List) -> np.ndarray:
        detections = []
        frame_height, frame_width, _ = frame.shape
        for detection in mp_detections:
            height = int(detection.location_data.relative_bounding_box.height * frame_height)
            width = int(detection.location_data.relative_bounding_box.width * frame_width)
            left = max(0 ,int(detection.location_data.relative_bounding_box.xmin * frame_width))
            top = max(0 ,int(detection.location_data.relative_bounding_box.ymin * frame_height))

            detections.append([top, left, top + height, left + width])

        return np.array(detections)

    def __call__(self, frame: np.ndarray, return_tlbr: bool = False) -> np.ndarray:

        results = self.face_detection.process(frame)

        if return_tlbr:
            if results.detections:
                return self.tlbr(frame, results.detections)
            return []

        if results.detections:
            for tlbr in self.tlbr(frame, results.detections):
                    cv2.rectangle(frame, tlbr[:2][::-1], tlbr[2:][::-1], self.color, self.thickness)


        
        if results.detections:
            return frame, tlbr[:1],tlbr[1:2],tlbr[2:3],tlbr[3:4]
        else:
            return frame, None,None,None,None