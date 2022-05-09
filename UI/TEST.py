import sys
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL._imaging import font
from PyQt5.QtCore import QUrl, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from numpy import random
import argparse
import os
import time
from pathlib import Path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
'''单张图片检测'''
class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()
        self.resize(1600, 900)
        self.setWindowTitle("test")

        self.label1 = QLabel(self)
        self.label1.setText("   待检测图片")
        self.label1.setFixedSize(700, 500)
        self.label1.move(110, 80)

        self.label2 = QLabel(self)
        self.label2.setText("   检测结果")
        self.label2.setFixedSize(700, 500)
        self.label2.move(850, 80)

        self.label3 = QLabel(self)
        self.label3.setText("")
        self.label3.move(1200, 620)
        self.label3.setStyleSheet("font-size:20px;")
        self.label3.adjustSize()
        self.timer_camera = QTimer()
        self.type = ''
        btn = QPushButton(self)
        btn.setText("打开图片")
        btn.move(10, 240)
        btn.clicked.connect(self.open_image)

        btn3 = QPushButton(self)
        btn3.setText("打开video(mp4格式)")
        btn3.move(10, 280)
        btn3.clicked.connect(self.getcap)

        btn1 = QPushButton(self)
        btn1.setText("检测图片")
        btn1.move(10, 200)
        btn1.clicked.connect(self.detectimg)

        btn2 = QPushButton(self)
        btn2.setText("视频检测")
        btn2.move(10, 160)
        btn2.clicked.connect(self.detectvideo)

        btn_Start = QPushButton(self)
        btn_Start.setText("开始播放")
        btn_Start.move(10, 360)
        btn_Start.clicked.connect(self.Btn_Start)

        btn_Stop = QPushButton(self)
        btn_Stop.setText("播放结束")
        btn_Stop.move(10, 400)
        btn_Stop.clicked.connect(self.Btn_Stop)
    def open_image(self):

        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        '''上面一行代码是弹出选择文件的对话框，第一个参数固定，第二个参数是打开后右上角显示的内容
            第三个参数是对话框显示时默认打开的目录，"." 代表程序运行目录
            第四个参数是限制可打开的文件类型。
            返回参数 imgName为G:/xxxx/xxx.jpg，imgType为*.jpg。	
            此时相当于获取到了文件地址 
        '''
        self.imgName = imgName
        imgName_cv2 = cv2.imread(imgName)
        im0 = cv2.cvtColor(imgName_cv2, cv2.COLOR_RGB2BGR)
        # 这里使用cv2把这张图片读取进来，也可以用QtGui.QPixmap方式。然后由于cv2读取的跟等下显示的RGB顺序不一样，所以需要转换一下顺序
        showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
        self.label1.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.label1.setScaledContents(True)
        # 然后这个时候就可以显示一张图片了。
    # def open_video(self):
    #     """选取视频文件"""
    #     videopath, videoType = QFileDialog.getOpenFileName(self, 'chose files', '', 'Image files(*.mp4 *.avi)')  # 打开文件选择框选择文件
    #     self.videopath = videopath
    #     file_name = os.path.basename(self.videopath).split('.')[0]
    #     file_type = os.path.basename(self.videopath).split('.')[1]
    #     cap = cv2.VideoCapture(videopath)
    #     isopen = cap.isOpened()
    #     while isopen:
    #         retval, image = cap.read()
    #         c = 1
    #         if retval and c % 10 == 0 :
    #             if len(image.shape) == 3:
    #                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #                 vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
    #             elif len(image.shape) == 1:
    #                 vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Indexed8)
    #             else:
    #                 vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
    #
    #             self.label1.setPixmap(QPixmap(vedio_img))
    #             self.label1.setScaledContents(True)  # 自适应窗口
    #             c+=1
    #         else:
    #             cap.release()
    #一下是获取视频
    def getcap(self):
        self.type = 'test'
        videopath, videoType = QFileDialog.getOpenFileName(self, 'chose files', '', 'Image files(*.mp4 *.avi)')
        self.imgName = videopath
        self.cap = cv2.VideoCapture(videopath)
    def OpenFrame(self):

        ret, frame = self.cap.read()

        if ret:
            # Process
            # cv2.putText(frame,"good",(50,100),2,cv2.FONT_HERSHEY_COMPLEX,(0,0,255),3)

            self.Display_Image(frame)
        else:
            self.cap.release()
            self.timer_camera.stop()

    def Display_Image(self, image):
        if (len(image.shape) == 3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Q_img = QImage(image.data,
                           image.shape[1],
                           image.shape[0],
                           QImage.Format_RGB888)
        elif (len(image.shape) == 1):
            Q_img = QImage(image.data,
                           image.shape[1],
                           image.shape[0],
                           QImage.Format_Indexed8)
        else:
            Q_img = QImage(image.data,
                           image.shape[1],
                           image.shape[0],
                           QImage.Format_RGB888)
        if self.type == 'res':
            self.label2.setPixmap(QtGui.QPixmap(Q_img))
            self.label2.setScaledContents(True)
        elif self.type == 'test':
            self.label1.setPixmap(QtGui.QPixmap(Q_img))
            self.label1.setScaledContents(True)

    def Btn_Close(self, event):
        event.accept()
        self.cap.release()

    def Btn_Start(self):
        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(30)
        self.timer_camera.timeout.connect(self.OpenFrame)

    def Btn_Stop(self):
        # self.cap.release()
        self.timer_camera.stop()

    def detectvideo(self):
        self.type = 'res'
        file_name = os.path.basename(self.imgName).split('.')[0]
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default=r'E:\repository\taishen-s222\Deeplearning\yolov5-5.0\runs\train\2000_None_3classes\weights\best.pt',
                            help='model.pt path(s)')
        parser.add_argument('--source', type=str,
                            default=self.imgName,
                            help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', default='False', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=r'E:\repository\taishen-s222\Deeplearning\yolov5-5.0\UI\detectimg',
                            help='save results to project/name')
        parser.add_argument('--name', default=file_name, help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        opt = parser.parse_args()
        print(opt)

        # check_requirements(exclude=('pycocotools', 'thop'))
        def detect(save_img=False):
            source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
            save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
            webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
                ('rtsp://', 'rtmp://', 'http://', 'https://'))

            # Directories
            save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Initialize
            set_logging()
            device = select_device(opt.device)
            half = device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size
            if half:
                model.half()  # to FP16

            # Second-stage classifier
            classify = False
            if classify:
                modelc = load_classifier(name='resnet101', n=2)  # initialize
                modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(
                    device).eval()

            # Set Dataloader
            vid_path, vid_writer = None, None
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            t0 = time.time()
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)
                t2 = time_synchronized()

                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # img.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + (
                        '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or view_img:  # Add bbox to image
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    # Print time (inference + NMS)
                    print(f'{s}Done. ({t2 - t1:.3f}s)')

                    # Stream results
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                             (w, h))
                            vid_writer.write(im0)
                        # 这里是画检测框的代码
                    labels = []
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        # 统计当前图片检测出来的label数量
                        labels.append(label)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        # 获取txt路径
                    print('txt_path:', txt_path)
                    print('save_path:', save_path)
                    # txt_path = p.replace('images', 'labels')
                    # txt_path= p.replace('jpg', 'txt')
                    # 获取ground truth的数量
                    try:
                        with open(txt_path + '.txt', 'r') as f:
                            num = f.readlines()
                        if len(num) != len(labels):
                            cv2.imwrite('runs/detect/worry/img', im0)
                        else:
                            pass
                    except:
                        pass

            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                print(f"Results saved to {save_dir}{s}")

            print(f'Done. ({time.time() - t0:.3f}s)')

        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    detect()
                    strip_optimizer(opt.weights)
            else:
                print('开始detect')
                detect()
        print('开始读取结果')
        while not os.path.exists(
                r'E:\repository\taishen-s222\Deeplearning\yolov5-5.0\UI\detectimg/' + file_name + '/' + os.path.basename(
                    self.imgName)):
            pass
        self.cap = cv2.VideoCapture(r'E:\repository\taishen-s222\Deeplearning\yolov5-5.0\UI\detectimg/' + file_name + '/' + os.path.basename(
                self.imgName))
        self.Btn_Start()
        # imgName_cv2 = cv2.imread(
        #     r'E:\repository\taishen-s222\Deeplearning\yolov5-5.0\UI\detectimg/' + file_name + '/' + os.path.basename(
        #         self.imgName))
        # im0 = cv2.cvtColor(imgName_cv2, cv2.COLOR_RGB2BGR)
        # # 这里使用cv2把这张图片读取进来，也可以用QtGui.QPixmap方式。然后由于cv2读取的跟等下显示的RGB顺序不一样，所以需要转换一下顺序
        # showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
        # self.label2.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # self.label2.setScaledContents(True)
        # 然后这个时候就可以显示一张图片了。

    def detectimg(self):

        file_name = os.path.basename(self.imgName).split('.')[0]
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default=r'E:\repository\taishen-s222\Deeplearning\yolov5-5.0\runs\train\2000_None_3classes\weights\best.pt',
                            help='model.pt path(s)')
        parser.add_argument('--source', type=str,
                            default=self.imgName,
                            help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', default='False', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=r'E:\repository\taishen-s222\Deeplearning\yolov5-5.0\UI\detectimg', help='save results to project/name')
        parser.add_argument('--name', default=file_name, help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        opt = parser.parse_args()
        print(opt)
        #check_requirements(exclude=('pycocotools', 'thop'))
        def detect(save_img=False):
            source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
            save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
            webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
                ('rtsp://', 'rtmp://', 'http://', 'https://'))

            # Directories
            save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Initialize
            set_logging()
            device = select_device(opt.device)
            half = device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size
            if half:
                model.half()  # to FP16

            # Second-stage classifier
            classify = False
            if classify:
                modelc = load_classifier(name='resnet101', n=2)  # initialize
                modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(
                    device).eval()

            # Set Dataloader
            vid_path, vid_writer = None, None
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            t0 = time.time()
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)
                t2 = time_synchronized()

                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # img.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + (
                        '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or view_img:  # Add bbox to image
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    # Print time (inference + NMS)
                    print(f'{s}Done. ({t2 - t1:.3f}s)')

                    # Stream results
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer.write(im0)
                        # 这里是画检测框的代码
                    labels = []
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        # 统计当前图片检测出来的label数量
                        labels.append(label)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        # 获取txt路径
                    print('txt_path:', txt_path)
                    print('save_path:', save_path)
                    # txt_path = p.replace('images', 'labels')
                    # txt_path= p.replace('jpg', 'txt')
                    # 获取ground truth的数量
                    try:
                        with open(txt_path + '.txt', 'r') as f:
                            num = f.readlines()
                        if len(num) != len(labels):
                            cv2.imwrite('runs/detect/worry/img', im0)
                        else:
                            pass
                    except:
                        pass

            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                print(f"Results saved to {save_dir}{s}")

            print(f'Done. ({time.time() - t0:.3f}s)')

        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    detect()
                    strip_optimizer(opt.weights)
            else:
                print('开始detect')
                detect()
        print('开始读取结果')
        while not os.path.exists(r'E:\repository\taishen-s222\Deeplearning\yolov5-5.0\UI\detectimg/'+file_name+'/'+os.path.basename(self.imgName)):
             pass
        imgName_cv2 = cv2.imread(r'E:\repository\taishen-s222\Deeplearning\yolov5-5.0\UI\detectimg/'+file_name+'/'+os.path.basename(self.imgName))
        im0 = cv2.cvtColor(imgName_cv2, cv2.COLOR_RGB2BGR)
        # 这里使用cv2把这张图片读取进来，也可以用QtGui.QPixmap方式。然后由于cv2读取的跟等下显示的RGB顺序不一样，所以需要转换一下顺序
        showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
        self.label2.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.label2.setScaledContents(True)
        # 然后这个时候就可以显示一张图片了。

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui_p = picture()
    ui_p.show()
    sys.exit(app.exec_())
