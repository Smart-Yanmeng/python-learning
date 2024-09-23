import threading
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QIcon, QColor
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import (
    QFileDialog,
    QWidget,
    QLabel,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QFormLayout,
    QPushButton,
    QLineEdit,
    QTabWidget,
    QMessageBox,
)

from PyQt5.QtGui import *
import face_recognition
import cv2


import os

from utils import *


# 窗口主类
class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化设置
        super().__init__()
        self.setWindowTitle("人脸识别系统")
        self.resize(1100, 650)
        self.setWindowIcon(QIcon("UI_images/logo.png"))
        # 要上传的图片路径
        self.up_img_name = ""
        # 要检测的图片名称
        self.input_fname = ""
        # 要检测的视频名称
        self.source = ""
        self.video_capture = cv2.VideoCapture(0)
        # 初始化中止事件
        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        # 初始化人脸向量
        self.known_names, self.known_encodings = self.initFaces()
        # 加载lbp检测器
        # 加载人脸识别模型
        # 初始化界面
        self.initUI()
        self.set_down()

    # 初始化数据库的人脸
    def initFaces(self):
        # 存储知道人名列表
        known_names = []
        # 存储知道的特征值
        known_encodings = []
        # 遍历存储人脸图片初始库的文件夹
        db_folder = "images/db_faces"
        face_imgs = os.listdir(db_folder)
        # 遍历图片，将人脸图片转化为向量
        for face_img in face_imgs:
            face_img_path = os.path.join(db_folder, face_img)
            face_name = face_img.split(".")[0]
            load_image = face_recognition.load_image_file(face_img_path)  # 加载图片
            image_face_encoding = face_recognition.face_encodings(load_image)[
                0
            ]  # 获得128维特征值
            known_names.append(face_name)  # 添加到人名的列表
            known_encodings.append(image_face_encoding)  # 添加到向量的列表
        return known_names, known_encodings

    # 初始化界面
    def initUI(self):
        # 设置字体
        font_v = QFont("楷体", 14)
        generally_font = QFont("楷体", 15)
        # 图片检测
        img_widget = QWidget()
        img_layout = QVBoxLayout()
        img_f_title = QLabel("上传人脸图像")  # 设置标题
        img_f_title.setAlignment(Qt.AlignCenter)  # 设置标题位置为居中
        img_f_title.setFont(QFont("楷体", 18))  # 设置标题字体大小
        # todo 要上传的人脸图像
        self.img_label_prompt = QLabel(
            "请输入人脸姓名："
        )  # 设置第一个界面上要显示的图片
        self.img_f_img = QLabel()  # 设置第一个界面上要显示的图片
        self.img_f_img.setPixmap(
            QPixmap("UI_images/renlian1.jpg")
        )  # 初始化要显示的图片
        self.img_f_img.setAlignment(Qt.AlignCenter)  # 设置图片居中
        self.face_name = QLineEdit()  # 设置当前图片对应的人名

        img_up_btn = QPushButton("选择身份证照片")  # 设置上传图片的按钮
        img_det_btn = QPushButton("上传照片")  # 设置开始上传的按钮
        img_up_btn.clicked.connect(self.up_img)  # 联系到相关函数
        img_det_btn.clicked.connect(self.up_db_img)  # 连接到相关函数
        # 设置组件的样式
        img_up_btn.setFont(generally_font)
        img_det_btn.setFont(generally_font)
        img_up_btn.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}"
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        img_det_btn.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}"
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        # 将组件添加到布局上，然后设置主要的widget为当前的布局
        img_layout.addWidget(img_f_title)
        img_layout.addWidget(self.img_f_img)

        img_layout2 = QHBoxLayout()
        self.face_name.setStyleSheet(
            """
            QLineEdit {
                border: 1px solid #A0A0A0; /* 边框宽度为1px，颜色为#A0A0A0 */
                border-radius: 3px; /* 边框圆角 */
                width: 100px; /* 宽度 */
                padding-left: 5px; /* 文本距离左边界有5px */
                background-color: #F2F2F2; /* 背景颜色 */
                color: #A0A0A0; /* 文本颜色 */
                selection-background-color: #A0A0A0; /* 选中文本的背景颜色 */
                selection-color: #F2F2F2; /* 选中文本的颜色 */
                font-family: "Microsoft YaHei"; /* 文本字体族 */
                font-size: 20pt; /* 文本字体大小 */
            }

            QLineEdit:hover { /* 鼠标悬浮在QLineEdit时的状态 */
                border: 1px solid #298DFF;
                border-radius: 3px;
                background-color: #F2F2F2;
                    color: #298DFF;
                    selection-background-color: #298DFF;
                selection-color: #F2F2F2;
            }


            """
        )  # 设置背景颜色

        self.img_label_prompt.setStyleSheet(
            """
            QLabel {               
                font-family: "Microsoft YaHei"; /* 文本字体族 */
                font-size: 20pt; /* 文本字体大小 */
                padding-left: 5px;
            }
            
            """
        )  # 设置背景颜色

        img_layout2.addWidget(self.img_label_prompt)
        img_layout2.addWidget(self.face_name)
        img_layout.addLayout(img_layout2)

        img_layout.addWidget(img_up_btn)
        img_layout.addWidget(img_det_btn)
        img_widget.setLayout(img_layout)

        """
        *** 4. 视频识别界面 ***
        """
        video_widget = QWidget()
        video_layout = QVBoxLayout()
        # 设置视频识别区的标题
        self.video_title2 = QLabel("视频识别区")
        self.video_title2.setFont(font_v)
        self.video_title2.setAlignment(Qt.AlignCenter)
        self.video_title2.setFont(font_v)
        # 设置显示的界面
        self.DisplayLabel = QLabel()
        self.DisplayLabel.setPixmap(QPixmap(""))
        self.btn_open_rsmtp = QPushButton("检测摄像头")
        self.btn_open_rsmtp.setFont(font_v)
        # 设置打开摄像头的按钮和样式
        self.btn_open_rsmtp.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}"
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        # 设置选择文件的的按钮和样式
        self.btn_open = QPushButton("开始识别（选择文件）")
        self.btn_open.setFont(font_v)
        self.btn_open.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}"
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        # 设置结束演示的按钮和样式
        self.btn_close = QPushButton("结束检测")
        self.btn_close.setFont(font_v)
        self.btn_close.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}"
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        # 将组件添加到布局上
        self.btn_open_rsmtp.clicked.connect(self.open_local)
        self.btn_open.clicked.connect(self.open)
        self.btn_close.clicked.connect(self.close)
        video_layout.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_title2)
        video_layout.addWidget(self.DisplayLabel)
        self.DisplayLabel.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.btn_open_rsmtp)
        video_layout.addWidget(self.btn_open)
        video_layout.addWidget(self.btn_close)
        video_widget.setLayout(video_layout)
        """
        *** 5. 关于界面 ***
        """
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel("欢迎使用人脸检测系统\n\n")  # todo 修改欢迎词语
        about_title.setFont(QFont("楷体", 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        # about_img.setPixmap(QPixmap("UI_images/qq.png"))
        about_img.setAlignment(Qt.AlignCenter)

        # label4.setText("<a href='https://oi.wiki/wiki/学习率的调整'>如何调整学习率</a>")
        label_super = QLabel()  # todo 更换作者信息
        label_super.setText("zongying@2022")
        label_super.setFont(QFont("楷体", 16))
        label_super.setOpenExternalLinks(True)
        # label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)
        # 分别添加子页面
        self.addTab(img_widget, "上传人脸")
        self.addTab(video_widget, "视频检测")
        self.addTab(about_widget, "关于")
        self.setTabIcon(0, QIcon("UI_images/图片.png"))
        # self.setTabIcon(1, QIcon('UI_images/图片.png'))
        self.setTabIcon(1, QIcon("UI_images/直播.png"))
        self.setTabIcon(2, QIcon("UI_images/logo_about.png"))

    # 第一个界面的函数
    def up_img(self):
        # 打开文件选择框
        openfile_name = QFileDialog.getOpenFileName(
            self, "选择文件", "", "Image files(*.jpg , *.png)"
        )
        # 获取上传的文件名称
        img_name = openfile_name[0]
        if img_name == "":
            pass
        else:
            # 上传之后显示并做归一化处理
            src_img = cv_imread(img_name)
            src_img_height = src_img.shape[0]
            src_img_width = src_img.shape[1]
            target_img_height = 400
            ratio = target_img_height / src_img_height
            target_img_width = int(src_img_width * ratio)
            # 将图片统一处理到高为400的图片，方便在界面上显示
            target_img = cv2.resize(src_img, (target_img_width, target_img_height))
            cv_imwrite("UI_images/tmp/toup.jpg", target_img)
            self.img_f_img.setPixmap(QPixmap("UI_images/tmp/toup.jpg"))
            self.up_img_name = "UI_images/tmp/toup.jpg"

    def up_db_img(self):
        # 首先判断该图像是否有一个人脸，多个人脸或者没有人脸都不行
        face_name = self.face_name.text()
        if face_name == "":
            QMessageBox.information(self, "不能为空", "请填写人脸姓名")
        else:
            load_image = face_recognition.load_image_file(self.up_img_name)  # 加载图片
            image_face_encoding = face_recognition.face_encodings(
                load_image
            )  # 获得128维特征值
            encoding_length = len(image_face_encoding)  # 获取人脸得数量
            if encoding_length == 0:  # 如果没有人脸，提示用户重新上传
                QMessageBox.information(self, "请重新上传", "当前图片没有发现人脸")
            elif encoding_length > 1:  # 如果人脸有多个，也提示用户重新上传
                QMessageBox.information(self, "请重新上传", "当前图片发现多张人脸")
            else:
                face_encoding = image_face_encoding[0]  # 获取解析得到得人脸数量
                img = cv_imread(self.up_img_name)  # 将上传得图片保存在db目录下
                img_path = face_name + ".jpg"
                cv_imwrite("images/db_faces/" + img_path, img)
                # 上传之后重新对字典进行处理
                self.known_names.append(face_name)
                self.known_encodings.append(face_encoding)
                QMessageBox.information(self, "上传成功", "数据已上传！")

    """
    ### 3. 视频识别相关功能 ### 
    """

    # 关闭事件 询问用户是否退出
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "退出",
            "是否要退出程序？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    # 读取录像文件
    def open(self):
        # 选择录像文件进行读取
        mp4_fileName, fileType = QFileDialog.getOpenFileName(
            self, "Choose file", "", "*.mp4"
        )
        if mp4_fileName:
            # 启动录像文件读取得线程并在画面上实时显示
            self.source = mp4_fileName
            self.video_capture = cv2.VideoCapture(self.source)
            th = threading.Thread(target=self.display_video)
            th.start()

    def open_local(self):
        # 选择录像文件进行读取
        mp4_filename = 0
        self.source = mp4_filename
        # 读取摄像头进行实时得显示
        self.video_capture = cv2.VideoCapture(self.source)
        th = threading.Thread(target=self.display_video)
        th.start()

    # 退出进程
    def close(self):
        # 点击关闭按钮后重新初始化界面
        self.stopEvent.set()
        self.set_down()

    # todo 执行人脸识别主进程
    def display_video(self):
        # 首先把打开按钮关闭
        self.btn_open.setEnabled(False)
        self.btn_close.setEnabled(True)
        process_this_frame = True
        while True:
            ret, frame = self.video_capture.read()  # 读取摄像头
            if frame is None:
                break
            # opencv的图像是BGR格式的，而我们需要是的RGB格式的，因此需要进行一个转换。
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )  # 将图像转化为rgb颜色通道
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 检查人脸 按照1.1倍放到 周围最小像素为5
            # face_zone = self.face_detect.detectMultiScale(gray_frame, scaleFactor=2, minNeighbors=2)  # maxSize = (55,55)
            if process_this_frame:
                face_locations = face_recognition.face_locations(
                    rgb_frame
                )  # 获得所有人脸位置
                face_encodings = face_recognition.face_encodings(
                    rgb_frame, face_locations
                )  # 获得人脸特征值
                face_names = []  # 存储出现在画面中人脸的名字
                for face_encoding in face_encodings:  # 和数据库人脸进行对比
                    # 如果当前人脸和数据库的人脸的相似度超过0.5，则认为人脸匹配
                    matches = face_recognition.compare_faces(
                        self.known_encodings, face_encoding, tolerance=0.5
                    )
                    if True in matches:
                        first_match_index = matches.index(True)
                        # 返回相似度最高的作为当前人脸的名称
                        name = self.known_names[first_match_index]
                    else:
                        name = "unknown"
                    face_names.append(name)
            process_this_frame = not process_this_frame
            # 将捕捉到的人脸显示出来
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(
                    frame, (left, top), (right, bottom), (0, 0, 255), 2
                )  # 画人脸矩形框
                # 加上人名标签
                # cv2.rectangle(
                #     frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
                # )
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(
                #     frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
                # )
                frame = draw_chinese_text(
                    frame, name, (left + 6, bottom - 6), 30, "white"
                )
            #
            # 保存图片并进行实时的显示
            # cv2.imshow("Video", frame)
            # cv2.waitKey(0)
            frame = frame
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]
            frame_scale = 500 / frame_height
            frame_resize = cv2.resize(
                frame, (int(frame_width * frame_scale), int(frame_height * frame_scale))
            )
            cv_imwrite("images/tmp.jpg", frame_resize)
            self.DisplayLabel.setPixmap(QPixmap("images/tmp.jpg"))
            if cv2.waitKey(30) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.DisplayLabel.clear()
                self.btn_close.setEnabled(False)
                self.btn_open.setEnabled(True)
                self.set_down()
                break
        self.btn_open.setEnabled(True)
        self.btn_close.setEnabled(False)
        self.set_down()

    # 初始化视频检测界面
    def set_down(self):
        self.video_capture.release()
        # zongying
        cv2.destroyAllWindows()
        self.DisplayLabel.setPixmap(QPixmap("UI_images/ae862.jpg"))


# https://www.lfd.uci.edu/~gohlke/pythonlibs/
# https://pypi.org/project/dlib/#files
if __name__ == "__main__":
    # 加载页面
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
