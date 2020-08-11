import cv2
import sys
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QVBoxLayout, QPushButton, QFileDialog, QMainWindow, \
    QLineEdit, QTextBrowser
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

import time
import numpy as np
import math

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 GUI - Social Distance Detector'
        self.left = 300
        self.top = 100
        self.width = 1000
        self.height = 850
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)


        # Label Input
        self.label = QLabel(self)
        self.label.setText("Input")
        # self.label.move(25, 35)
        self.label.move(245, 35)
        self.label.resize(280, 40)

        # Label Output
        self.label = QLabel(self)
        self.label.setText("Output")
        self.label.move(25, 135)
        # self.label.resize(280, 40)

        #Text browser
        self.txtbrowser = QTextBrowser(self)
        # self.txtbrowser.move(80, 35)
        self.txtbrowser.move(300, 35)
        self.txtbrowser.resize(300, 40)


        #Label that holds the image
        self.disply_width = 900
        self.display_height = 950
        self.image_label = QLabel(self)
        self.image_label.move(50, 30)
        self.image_label.resize(self.disply_width, self.display_height)

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # BTN for webcam
        self.button_webcam = QPushButton('Using Webcam', self)
        # self.button_webcam.move(400, 55)
        self.button_webcam.move(620, 55)
        self.button_webcam.clicked.connect(self.btn_webcam)

        # BTN for thread (Start)
        self.button_th = QPushButton('Start', self)
        self.button_th.move(360, 130)
        self.button_th.clicked.connect(self.btn_th)

        #BTN for STOP
        self.button_stop = QPushButton('Stop', self)
        self.button_stop.move(500, 130)
        self.button_stop.clicked.connect(self.btn_stop)



        #BTN for Browser
        self.button = QPushButton('Browser', self)
        # self.button.move(400, 20)
        self.button.move(620, 20)

        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.show()

    #Start btn
    def btn_th(initUI):

        initUI.thread = VideoThread()
        # connect its signal to the update_image slot
        initUI.thread.change_pixmap_signal.connect(initUI.update_image)
        # start the thread
        initUI.thread.start()

    #STOP btn
    def btn_stop(initUI):
        initUI.thread.stop()
        initUI.image_label.clear()


    def btn_webcam(initUI):
        webcam_src = 0
        initUI.txtbrowser.setText("Using Webcam")
        # initUI.label.adjustSize()

        f = open("path.txt", "w")
        f.write('%d' % webcam_src)
        f.close()


    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, frame):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(frame)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, frame):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        # convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


    @pyqtSlot()
    def on_click(self):

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        # fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
        #                                           "All Files (*);;Python Files (*.py)", options=options)
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose video file", "",
                                                  "MP4 (*.mp4);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)
        self.txtbrowser.setText(fileName)
        # self.txtbrowser.adjustSize()

        f = open("path.txt", "w")
        f.write(fileName)
        f.close()



#Lay ra output layers name tu Yolo
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Ham ve cac hinh chu nhat va ten class
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    # color = COLORS[class_id]
    color = (0,255,0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # centerPoint = (int(x+w/2),int(y+h/2))
    # dot = cv2.circle(img, centerPoint, 7, (0, 0, 255), -1)
# color = (0,255,0)
personID = 0

writer = None
# cap = cv2.VideoCapture(args["video"] if args["video"] else 1)
(W,H) = (None, None)

# Doc ten cac class
classes = None
with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

#Khoi tao gia tri mau cho class
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
#Load  yolo qua dnn
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')


nCount = 0

starting_time = time.time()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        frame_id = 0
        nCount= 0
        (W, H) = (None, None)

        f = open("path.txt", "r")
        plaintext = f.read()
        if plaintext == '0':
            print('Using Webcam')
            cap1 = cv2.VideoCapture(int(plaintext))
        else:
            print('Using video path')
            cap1 = cv2.VideoCapture(plaintext)
        while self._run_flag:
            ret, frame = cap1.read()
            if ret:
                frame_id += 1

                # frame khong grabbed: end video stream
                if not ret:
                    break
                #Load input image and extracts dimentions
                if W is None or H is None:
                    (H, W) = frame.shape[:2]

                #xay dung blob tu input image, truyen tham so vao yolo detector
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                outs = net.forward(get_output_layers(net))

                # Khoi tao cac list
                class_ids = []
                confidences = []
                boxes = []
                #threshold loc cac confidence yeu
                conf_threshold = 0.5
                #threshold cho NMS
                nms_threshold = 0.3

                #Loop qua moi layer outputs
                for out in outs:
                    #Loop qua moi detection
                    for detection in out:
                        #extract confidence & class
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if (confidence > conf_threshold) and (class_id == personID):
                            #scale lai toa do bounding box theo size image
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")
                            #su dung tdo(x,y) de tinh top and left corner cua bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            centerPoint = (int(x + width / 2), int(y + height / 2))

                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([x, y, int(width), int(height), centerPoint])

                #apply NMS
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                # Ve cac khung chu nhat quanh doi tuong
                for i in indices:
                    i = i[0]
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    draw_prediction(frame, class_ids[i], round(x), round(y), round(x + w), round(y + h))
                    confiNum = "%.3f" % confidences[i]
                    cv2.putText(frame, str(confiNum),
                                (x - 10, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

                    #List of boundingbox's center
                    centerList = [item[4] for item in boxes]

                    ########Print indices of "Person" in class_ids
                    indices_person = [i for i, x in enumerate(class_ids) if x == 0]

                    for i in range(len(indices_person)):
                        ##Coordiante of person.
                        # print(centerList[i])
                        # print(centerList[i][0])
                        for j in range(i + 1, len(indices_person)):

                            cv2.line(frame, centerList[i], centerList[i + 1], (255, 0, 0), 2, 8)
                            #Eucliden distance
                            distance = math.sqrt(((centerList[i][0] - centerList[i + 1][0]) ** 2) + (
                                        (centerList[i][1] - centerList[i + 1][1]) ** 2))
                            cv2.putText(frame, (str(distance)).split('.')[0],
                                        (centerList[i][0] - 10, centerList[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 0, 0), 2)

                            if distance <= 150:

                                # if nCount == 1 and distance <= 30:
                                #     passed = True
                                # while(passed == False):
                                cv2.putText(frame, "Warning", (27, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
                                cv2.putText(frame, "Warning", (centerList[i][0] - 20, centerList[i][1] - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                cv2.line(frame, centerList[i], centerList[i + 1], (0, 0, 255), 2, 8)
                    try:
                        distance
                        print(distance)
                    except NameError:
                        distance = None
                    # print(distance)

                elapsed_time = time.time() - starting_time
                fps = frame_id / elapsed_time
                cv2.putText(frame, "FPS: " + str(round(fps, 2)), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
                self.change_pixmap_signal.emit(frame)
        # shut down capture system
        cap1.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


if __name__ == '__main__':
    while True:
        app = QApplication(sys.argv)
        ex = App()
        ex.show()
        sys.exit(app.exec_())