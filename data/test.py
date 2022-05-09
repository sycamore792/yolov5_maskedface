import os


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

