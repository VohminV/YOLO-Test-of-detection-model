import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

class VideoProcessor(QWidget):
    CLASS_COLORS = {
        'название класса': "цвет класса",
    }

    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("YOLO Video Detection")
        self.setGeometry(100, 100, 640, 700)
        
        self.layout = QVBoxLayout()
        
        self.btn_select = QPushButton("Выбрать видео")
        self.btn_select.setFixedHeight(30)
        self.btn_select.clicked.connect(self.select_video)
        self.layout.addWidget(self.btn_select)

        self.btn_stop = QPushButton("Остановить видео")
        self.btn_stop.setFixedHeight(30)
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setEnabled(False)  
        self.layout.addWidget(self.btn_stop)
        
        self.label_video = QLabel()
        self.layout.addWidget(self.label_video)
        
        self.setLayout(self.layout)
        
        self.video_path = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        
        self.model = YOLO("МОДЕЛЬ")  
        self.conf_threshold = 0.50  

    def select_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Выберите видеофайл", "", "Видео файлы (*.mp4 *.avi *.mov)")
        
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.timer.start(int(1000 / self.fps))
            self.btn_stop.setEnabled(True)  

    def stop_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.label_video.clear()
        self.btn_stop.setEnabled(False)  
        print("Воспроизведение остановлено.")

    def process_frame(self):
        if self.cap and self.cap.isOpened():
            self.cap.grab()  
            ret, frame = self.cap.read()
            if not ret:
                self.stop_video()
                return
            
            frame_resized = cv2.resize(frame, (640, 640))
            results = self.model.predict(frame_resized)  
            
            inference_time = results[0].speed['inference']  
            img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            detection_count = 0
            object_centers = []

            for result in results:
                for box in result.boxes:
                    confidence = box.conf[0].item()
                    if confidence >= self.conf_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_name = result.names[int(box.cls[0])]
                        
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        object_centers.append((center_x, center_y, class_name, confidence))  
                        
                        detection_count += 1
            
            for (cx, cy, class_name, confidence) in object_centers:
                color = self.CLASS_COLORS.get(class_name, "white")
                
                # Рисуем крестик
                draw.line([(cx - 10, cy), (cx + 10, cy)], fill=color, width=3)
                draw.line([(cx, cy - 10), (cx, cy + 10)], fill=color, width=3)

                # Добавляем текст с классом и вероятностью
                text = f"{class_name} {confidence:.1%}"
                text_position = (cx - 30, cy - 20)  # Смещаем вверх
                draw.text(text_position, text, fill=color)

            frame_final = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            h, w, ch = frame_final.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_final.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label_video.setPixmap(QPixmap.fromImage(q_img))
            
            avg_inference_time = 95
            adjusted_delay = max(1, int(avg_inference_time - inference_time))
            self.timer.setInterval(adjusted_delay)
            
            print(f"Inference time: {inference_time:.1f}ms (adjusted delay: {adjusted_delay}ms), Detections: {detection_count}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessor()
    window.show()
    sys.exit(app.exec_())
