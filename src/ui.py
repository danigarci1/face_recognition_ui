import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt  # Add this import at the beginning of your file
import cv2
from faceDetector import face_detector
from faceRecognition import face_recognition
from ultralytics import YOLO
from utils.draw import drawPerson
import torch
from save_encodings import SaveEncodings
class ImageApp(QMainWindow):
    recognition_confidence = 0.2
    recognition_model_path = "assets/models/facenet512_weights.onnx"
    model_path = "assets/models/onlyface.pt"
    detection_confidence = 0.4
    def __init__(self):
        super().__init__()
        self.setWindowTitle('pyFRec')
        self.setGeometry(100, 100, 800, 600)

        # The label that displays the image
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(400, 400)

        # 'Load Image' button
        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.load_image)

        # 'Generate Embe' button
        self.generate_button = QPushButton('Generate Embeding', self)
        self.generate_button.clicked.connect(self.generate_embe)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.generate_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not file_name:
            return
        cv_image = cv2.imread(file_name)
            
        detector = face_detector.FaceDetection()
        recog = face_recognition.FaceNet(
            detector=detector,
            threshold=self.recognition_confidence,
            onnx_model_path = self.recognition_model_path)
        yolov8_detector = YOLO(self.model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        yolov8_detector.to(device)
        # Capture frame-by-frame
        # if the input queue *is* empty, give the current frame to
        # classify

        result = yolov8_detector.track(cv_image,verbose=False,persist=True,conf=self.detection_confidence)[0] #Verbose False to avoid yolov8 messages

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
                person_frame = cv_image[bbox_top:bbox_h,bbox_left:bbox_w,:]
                name = recog(frame=person_frame,face_frame=True)

            drawPerson(cv_image,bbox_left,bbox_top,bbox_w,bbox_h,name,(0,0,255))  

        # Convert the image from BGR to RGB
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Get the dimensions of the image for the QImage
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        
        # Create a QImage from the OpenCV image data
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(q_image)
        
        # Set the pixmap for the label
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))


    def generate_embe(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.run_embe_function(folder_path)

    def run_embe_function(self, folder_path):
        # Placeholder for the function to be called
        print(f"Function to process folder: {folder_path}")
        det = face_detector.FaceDetection()
        x = SaveEncodings(detector=det,faces_path=folder_path)
        x()


def main():
    from PyQt5.QtGui import QPalette, QColor
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Now use a palette to switch to dark colors:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    ex = ImageApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
