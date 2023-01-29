import torch
from joblib import delayed,Parallel
from torchvision.datasets import ImageFolder
import os
import torchvision
import torchvision.transforms as transforms

import numpy as np
# import numpy as np
import cv2
import random
from tqdm import tqdm_notebook
# calculate means and std
# train_txt_path = './train_val_list.txt'
# CNum = 10000 # 挑选多少图片进行计算
#
# imgs = np.zeros([1,3,100,100,100])
# means, stdevs = [], []
# root='/jhcnas1/zhoutaichang/original/'
# files=os.listdir(root)
# for i in tqdm_notebook(range(len(files))):
#   img_path = root+files[i]
#   img = torch.load(img_path).cpu().numpy()
#   # img = cv2.resize(img, (img_h, img_w))
#   img = img[np.newaxis,:,:, :, :]
#   imgs = np.concatenate((imgs, img), axis=0)
#       # print(i)
# imgs = imgs.astype(np.float32)/255.
# for i in tqdm_notebook(range(3)):
#       pixels = imgs[:,:,i,:].ravel() # 拉成一行
#       means.append(np.mean(pixels))
#       stdevs.append(np.std(pixels))
# # cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
# means.reverse() # BGR --> RGB
# stdevs.reverse()
# print("normMean = {}".format(means))
# print("normStd = {}".format(stdevs))
# print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

def prepare(root,j:int):
    i=os.listdir(root)
    return torch.load(root+i[j]).cpu().detach().numpy()
mean = torch.zeros(3)
std = torch.zeros(3)
loader=[]
def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    # print(len(loader))
    # train_loader = torch.utils.data.DataLoader(
        # train_data, batch_size=1, shuffle=False, num_workers=0,
        # pin_memory=True)
    global loader
    loader=Parallel(n_jobs=20,backend='threading')(delayed(prepare)(train_data,j)for j in range(len(os.listdir(train_data))))
    global mean
    global std
    loader=np.array(loader)
    print(loader[0].shape)
    for X in loader:
        for d in range(3):
            mean[d] += X[d, :, :, :].mean()
            std[d] += X[d, :, :, :].std()
    mean.div_(len(loader))
    std.div_(len(loader))

    return list(mean.numpy()), list(std.numpy()),loader


if __name__ == '__main__':
    train_dataset = '/jhcnas1/zhoutaichang/original/'
    m,s,l=getStat(train_dataset)
    print(m,s)
    a=transforms.Normalize(m,s)
    print(l[0,1,0,:10,:10])
    l = np.moveaxis(l, 0, -1)
    l=torch.from_numpy(l)

    img=a(l)
    print(img[0,1,0,:10,:10])
