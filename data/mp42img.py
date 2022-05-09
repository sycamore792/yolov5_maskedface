# 视频分解图片
# 1 load 2 info 3 parse 4 imshow imwrite
import os

import cv2

class mp42img():
    def __init__(self,mp4path,savepath,timeF):
        self.mp4path = mp4path
        self.savepath = savepath
        self.timeF = timeF
        self.cap = cv2.VideoCapture(mp4path)
        self.isOpened = self.cap.isOpened
    def getimg(self):
        c = 1
        while (self.isOpened):
            rval, frame = self.cap.read()
            if not rval:
                break
            if c % self.timeF == 0:  # 每隔timeF帧进行存储操作
                cv2.imwrite('mp42img/' + str(c / self.timeF) + '.jpg', frame)  # 存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
            c += 1
            cv2.waitKey(1)

        print('end!')

def get_class_nc(path):
    files = os.listdir(path)
    dic = {'0':0,'1':0,'2':0}
    for i in files:
        if '.txt' in i:
            with open(path+'/'+i,'r') as f:
                line = f.readlines()
                for x in line:
                    dic[x[0]]+=1
    return print('dic:',dic)

if __name__ == '__main__':
    #a = mp42img(r'F:\WIN_20220507_14_20_03_Pro.mp4','mp42img',40)
   # a.getimg()
    get_class_nc(r'E:\repository\taishen-s222\Deeplearning\data\labels_mp42img')
    get_class_nc(r'E:\repository\taishen-s222\Deeplearning\data\labels_mp42img1')
    get_class_nc(r'E:\repository\taishen-s222\Deeplearning\data\labels_mp42img2')
    #255，90，306