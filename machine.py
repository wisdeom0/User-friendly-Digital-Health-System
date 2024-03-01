from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel, QDialog, QVBoxLayout, QDesktopWidget, QComboBox
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QImage, QFont, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import sys
import os
from For_Project import weight_save, Button, Pose_Correct, Sensor
import csv
import cv2
from multiprocessing import Process, Pipe
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH") #qt plugin prob sloved!
imgpath_4 = r"/home/jeon/maker/image/배경4-1.png"
Exercise = "Back"
next_path  = r"C:/home/jeon/maker/image/다음버튼.png"
reset_path = r"C:/home/jeon/maker/image/종료버튼.png"
check_path = r"C:/home/jeon/maker/image/확인버튼.png"
correct_path = r"C:/home/jeon/maker/image/교정.png"




class WorkerThread(QThread):
    result_signal = pyqtSignal(str)

    def run(self):
        uid = Sensor.read_rfid()
        text = Sensor.rfid(uid)
        self.result_signal.emit(text)

class UserInputWindow(QDialog):
    rfid_recognized = pyqtSignal(str)  # Define a signal to indicate RFID recognition

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle('사용자 정보 입력')
        self.setFixedSize(800, 480)  # 800x480 크기로 고정

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(100, 149, 237))  # 배경색 설정
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))  # 텍스트 색상 설정
        self.setPalette(palette)

        label = QLabel('사용자의 정보를 입력하세요', self)
        label.setFont(QFont('Arial', 12, QFont.Bold))
        label.move(300, 150)  # QLabel의 위치 설정 (x, y 좌표)

        self.user_input = QLineEdit(self)
        self.user_input.move(315, 180)  # QLineEdit의 위치 설정 (x, y 좌표)

        self.ok_button = QPushButton('확인', self)
        self.ok_button.move(345, 250)  # QPushButton의 위치 설정 (x, y 좌표)
        self.ok_button.clicked.connect(self.accept)

        self.worker_thread = WorkerThread()
        self.worker_thread.result_signal.connect(self.rfid)
        self.worker_thread.start()

    def accept(self):
        self.user_data = self.user_input.text()
        super().accept()

    def rfid(self, text):
        self.user_input.setText(text)  # Update the text in QLineEdit
        self.rfid_recognized.emit(text)  # Emit the signal when RFID



csv_route = r"/home/jeon/maker/csv"
def run_another_task(pipe_conn):
    print("Another task is running...")
    while True:
        if pipe_conn.poll():
            frame = pipe_conn.recv() #result not receiving
            exe_gui = '랫풀 다운'
            Pose_Correct.pose_correction(exe_gui, csv_route, pipe_conn)
            result = Pose_Correct.pose_correction(exe_gui, csv_route, pipe_conn)
            pipe_conn.send(result) #child_conn send result? # gpt - send the result back to the main process
        pass


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.correction_labels = []
        self.screen = QDesktopWidget().screenGeometry()
        self.initUserInput()
        self.initUI()
        self.parent_conn, self.child_conn = Pipe()
        self.process = Process(target=run_another_task, args=(self.child_conn,))
        #self.result = self.parent_conn.recv() #parent_conn receive #why???????????????
        self.process.start()
        self.result = None


        self.initCamera()

    def initUserInput(self):
        self.user_input_window = UserInputWindow(self)
        if self.user_input_window.exec_():
            self.user_info = self.user_input_window.user_data
            if not self.user_info == None:
                self.last_info = weight_save.last_row(self.user_info, Exercise)
                with open(self.user_info + "_routine.csv", mode='r', newline='') as f:
                    reader = csv.reader(f)
                    self.routine_names = list(reader)
                    self.routine_names = self.routine_names[0]
                    print(self.routine_names)
            else:
                self.routine_names = None

    def initUI(self):
        print(self.user_info)
        image = QImage()
        image.load(imgpath_4)
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(self.screen.width(), self.screen.height(), Qt.KeepAspectRatio)

        self.resize(self.screen.width(), self.screen.height())
        self.image_path = imgpath_4
        self.font = QFont()
        self.font.setPointSize(7)
        self.resize(800, 480)
        self.show_weight()
        screen = QDesktopWidget().screenGeometry()

        self.setGeometry(0, 0, screen.width(), screen.height())

        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pixmap))
        self.setPalette(palette)
        for i, routine in enumerate(self.routine_names):
            frame = Button.ColorFrame(self, 20, 215 + 30*i, 160, 25, 'lightblue')
            label = Button.Set_Label(self, 32, 225 + 30*i, self.font, routine)
        self.label = Button.Set_Label(self, 400, 30, self.font, "횟수")
        self.label = Button.Set_Label(self, 595, 30, self.font, "무게")
        self.label = Button.Set_Label(self, 300, 30, self.font, f"{self.user_info}")
        self.reps_label = Button.Set_Spinbox(self, 420, 30, 0, 100, 1, self.font, 150, 27)
        self.weight_label = Button.Set_Spinbox(self, 615, 30, 0, 200, 5, self.font, 150, 27)

        self.next_button = Button.HoverButton(
            '다음', self, 620, 370, 40, 30, 'lightblue', 'gray', 'black', 'black')
        self.end_button = Button.HoverButton(
            '종료', self, 670, 370, 40, 30, 'pink', 'gray', 'black', 'black')
        self.correction_box = Button.ColorFrame(self, 240, 320, 300, 100, 'lightblue') #아래 파란박스

        self.next_button.clicked.connect(self.next_set_clicked)
        self.end_button.clicked.connect(self.reset)
        self.pose_correction()
        self.setWindowTitle('사용자의 정보를 입력하세요')
        self.camera_label = QLabel(self)
        self.camera_label.setGeometry(240, 80, 500, 235)  #camera 화면 위치
        #original 500 116 235 235

        self.show()

        self.pose_correction_timer = QTimer(self)
        self.pose_correction_timer.timeout.connect(self.pose_correction)
        self.pose_correction_timer.start(30)
    def initCamera(self):
        self.cap = cv2.VideoCapture(0)
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.updateCameraView)
        self.camera_timer.start(30)

    def updateCameraView(self):
        ret, frame = self.cap.read()
        #self.result = self.parent_conn.recv() #parent_conn receive
        #self.result = self.parent_conn.recv() #try... -> task stopped
        if self.result is not None:
            mp_drawing.draw_landmarks(frame, self.result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        #with mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.3) as pose:
        #    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #    image.flags.writeable = False
        #    results = pose.process(image)
        #    image.flags.writeable = True
        #    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        #    if results.pose_landmarks:
        #        print(results.pose_landmarks)
        #        for poselandmark in results.pose_landmarks.landmark:
        #            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if ret:
            frame = cv2.resize(frame, (416, 346))
            self.parent_conn.send(frame) # not sending ((frame,self.result))

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.camera_label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            print('Mouse Position:', event.x(), event.y())

    def reset(self):
        weight_save.input_and_save(self.user_info, Exercise,
            self.weight_label.value(), self.reps_label.value())
        try :
            os.remove("/home/jeon/maker/csv/jsonoutput.csv")
        except :
            pass
        self.close()
        self.new_window = MainWindow()
        self.new_window.show()

    def next_set_clicked(self):
        weight_save.input_and_save(self.user_info, Exercise,
            self.weight_label.value(), self.reps_label.value())
        self.weight_label.setValue(0)
        self.reps_label.setValue(0)
        try :
            os.remove("/home/jeon/maker/csv/jsonoutput.csv")
        except :
            pass
        self.show_weight()

    def recommending_weight(self):
        filename = f"{self.user_info}.csv"
        if not os.path.exists(filename):
            return [5, 5, 5]
        else :
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                data = list(reader)
                for row in reversed(data):
                    if row[1] == Exercise:
                        return row[-3:]
            return [5, 5, 5]

    def show_weight(self):
        self.machine_name = Button.Set_Label(self, 45, 100, self.font, Exercise)
        rm = self.recommending_weight()
        self.rm_1 = Button.Set_Label(self, 45, 135, self.font, f"1회 반복 무게 :{rm[0]}")
        self.rm_5 = Button.Set_Label(self, 45, 155, self.font, f"7~10회 반복 무게 :{rm[1]}")
        self.rm_10 = Button.Set_Label(self, 45, 175, self.font, f"11~15회 반복 무게 :{rm[2]}")




    def pose_correction(self):
        for label in self.correction_labels:
            label.deleteLater()

        self.correction_labels.clear()

        correction_list = []
        try:
            with open('/home/jeon/maker/csv/jsonoutput.csv', 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    correction_list = row[:-1]
        except:
            correction_list = None

        if correction_list is not None:
            for i, correction in enumerate(correction_list):
                label = QLabel(correction, self)
                label.setFont(self.font)

                if i % 2:
                    label.move(430, 325 + (i - 1) * 8)
                else:
                    label.move(270, 325 + i * 8)
                label.show()
                self.correction_labels.append(label)

        else:
            label = QLabel("Wait...", self)
            label.setFont(self.font)
            label.move(333, 400)
            label.show()
            self.correction_labels.append(label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())