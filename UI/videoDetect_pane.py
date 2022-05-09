from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimediaWidgets import QVideoWidget
from videoDetect import Ui_MainWindow
from myVideoWidget import myVideoWidget
import sys
from PyQt5 import QtCore, QtGui


class videoDetectionPane(Ui_MainWindow, QMainWindow):
    returnHome_signal = pyqtSignal()

    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("基层公安绩效评价工具 视频检测")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("resource/images/police.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.setIconSize(QtCore.QSize(40, 40))

        self.sld_video_pressed = False  # 判断当前进度条识别否被鼠标点击
        self.videoFullScreen = False  # 判断当前widget是否全屏
        self.videoFullScreenWidget = myVideoWidget()  # 创建一个全屏的widget

        self.player1 = QMediaPlayer()
        self.player2 = QMediaPlayer()

        self.player1.setVideoOutput(self.wgt_video)  # 视频播放输出的widget，就是上面定义的
        self.player2.setVideoOutput(self.wgt_video_2)

        self.btn_open.clicked.connect(self.openVideoFile)  # 打开视频文件按钮
        self.btn_play.clicked.connect(self.playVideo)  # play
        self.btn_stop.clicked.connect(self.pauseVideo)  # pause
        self.btn_cast.clicked.connect(self.castVideo)  # 视频截图
        self.player1.positionChanged.connect(self.changeSlide)  # change Slide
        self.player2.positionChanged.connect(self.changeSlide)
        self.videoFullScreenWidget.doubleClickedItem.connect(self.videoDoubleClicked)  # 双击响应
        self.wgt_video.doubleClickedItem.connect(self.videoDoubleClicked)  # 双击响应
        self.sld_video.setTracking(False)
        self.sld_video.sliderReleased.connect(self.releaseSlider)
        self.sld_video.sliderPressed.connect(self.pressSlider)
        self.sld_video.sliderMoved.connect(self.moveSlider)  # 进度条拖拽跳转
        self.sld_video.ClickedValue.connect(self.clickedSlider)  # 进度条点击跳转
        self.sld_audio.valueChanged.connect(self.volumeChange)  # 控制声音播放
        self.pushButton.clicked.connect(self.returnHome)  # 返回首页
        self.btn_cast.hide()

        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("resource/images/播放1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_play.setIcon(icon1)
        self.btn_play.setIconSize(QtCore.QSize(30, 30))

        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("resource/images/选择.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_open.setIcon(icon2)
        self.btn_open.setIconSize(QtCore.QSize(30, 30))

        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("resource/images/返回首页.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon4)
        self.pushButton.setIconSize(QtCore.QSize(30, 30))

        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("resource/images/暂停.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_stop.setIcon(icon3)
        self.btn_stop.setIconSize(QtCore.QSize(30, 30))

    def castVideo(self):
        screen = QGuiApplication.primaryScreen()
        cast_jpg = './' + QDateTime.currentDateTime().toString("yyyy-MM-dd hh-mm-ss-zzz") + '.jpg'
        screen.grabWindow(self.wgt_video.winId()).save(cast_jpg)

    def volumeChange(self, position):
        volume = round(position / self.sld_audio.maximum() * 100)  # 音量
        # print("vlume %f" %volume)
        self.player1.setVolume(volume)
        self.player2.setVolume(volume)
        self.lab_audio.setText("volume:" + str(volume) + "%")

    def clickedSlider(self, position):
        if self.player1.duration() > 0:  # 开始播放后才允许进行跳转
            video_position = int((position / 100) * self.player1.duration())
            self.player1.setPosition(video_position)
            self.lab_video.setText("%.2f%%" % position)
        else:
            self.sld_video.setValue(0)
        if self.player2.duration() > 0:  # 开始播放后才允许进行跳转
            video_position = int((position / 100) * self.player2.duration())
            self.player2.setPosition(video_position)
            self.lab_video.setText("%.2f%%" % position)
        else:
            self.sld_video.setValue(0)

    def moveSlider(self, position):
        self.sld_video_pressed = True
        if self.player1.duration() > 0:  # 开始播放后才允许进行跳转
            video_position = int((position / 100) * self.player1.duration())
            self.player1.setPosition(video_position)
            self.lab_video.setText("%.2f%%" % position)
        if self.player2.duration() > 0:  # 开始播放后才允许进行跳转
            video_position = int((position / 100) * self.player2.duration())
            self.player2.setPosition(video_position)
            self.lab_video.setText("%.2f%%" % position)

    def pressSlider(self):
        self.sld_video_pressed = True
        print("pressed")

    def releaseSlider(self):
        self.sld_video_pressed = False

    def changeSlide(self, position):
        if not self.sld_video_pressed:  # 进度条被鼠标点击时不更新
            self.vidoeLength = self.player1.duration() + 0.1
            self.sld_video.setValue(round((position / self.vidoeLength) * 100))
            self.lab_video.setText("%.2f%%" % ((position / self.vidoeLength) * 100))

    def openVideoFile(self):
        self.player1.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))  # 选取视频文件
        self.player1.play()  # 播放视频

        self.player2.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))
        self.player2.play()

        #print(self.player1.availableMetaData())
        #print(self.player2.availableMetaData())

    def playVideo(self):
        self.player1.play()
        self.player2.play()

    def pauseVideo(self):
        self.player1.pause()
        self.player2.pause()

    def videoDoubleClicked(self, text):
        if self.player1.duration() > 0:  # 开始播放后才允许进行全屏操作
            if self.videoFullScreen:
                self.player1.setVideoOutput(self.wgt_video)
                self.player2.setVideoOutput(self.wgt_video_2)
                self.videoFullScreenWidget.hide()
                self.videoFullScreen = False
            else:
                self.videoFullScreenWidget.show()
                self.player1.setVideoOutput(self.videoFullScreenWidget)
                self.player2.setVideoOutput(self.videoFullScreenWidget)
                self.videoFullScreenWidget.setFullScreen(1)
                self.videoFullScreen = True

    def returnHome(self):
        self.returnHome_signal.emit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    vieo_gui = videoDetectionPane()
    vieo_gui.show()
    sys.exit(app.exec_())