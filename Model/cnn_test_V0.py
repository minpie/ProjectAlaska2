import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

import os
import sys
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection


meta_info_path = "/storages/hdd0/pt0/ProjectAlaska/Datasets/track_a_learn_label.csv"
data_path = ""
data_path = "/storages/hdd0/pt0/ProjectAlaska/Datasets/output_images/"

tm = time.localtime(time.time())
print("Start at: " + time.strftime("%c", tm))





df = pd.read_csv(meta_info_path)
df["filename"] = df["filename"].apply(str)


dirs = df["type"].drop_duplicates().tolist()


data, label = [], []


for i, d in enumerate(dirs):
    print(i, "and", d)


    files = os.listdir(data_path)[:100]


    for f in files:
        #이미지 오픈
        img = Image.open(data_path + f, "r")
        #이미지 라벨 찾기
        find_label = df["type"].loc[df["filename"] == f.replace(".jpg", "")].to_string(index=False)
        # 이미지를 128,128로 리사이즈
        resize_img = img.resize((128,128))

        '''
        # 이미지를 RGB 컬러로 각각 쪼갠다
        # https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.split 참조
        r, g, b = resize_img.split()
        # 각 쪼갠 이미지를 255로 나눠서 0~1 사이의 값이 나오도록 정규화 한다.
        r_resize_img = np.asarray(np.float32(r) / 255.0)
        b_resize_img = np.asarray(np.float32(g) / 255.0)
        g_resize_img = np.asarray(np.float32(b) / 255.0)


        rgb_resize_img = np.asarray([r_resize_img, b_resize_img, g_resize_img])
        '''
        #'''
        bw = resize_img
        bw_resize_img = np.asarray(np.float32(bw) / 255.0)
        bw_resize_img = np.asarray([bw_resize_img])
        #'''




        # 이렇게 가공한 이미지를 추가한다.
        data.append(bw_resize_img)
        # data.append(f)


        # 라벨
        dirs_num = list(enumerate(dirs))
        dirs_num = dict(dirs_num)

        i = [k for k, v in dirs_num.items() if v == find_label][0]
        # 라벨 (0, hIVJ 1, EQbM 2, BghB 3, PRoU 4, nqBD 5, wbxA 6, KOPU 7, pvpk 8, aMyf)
        label.append(i)






# 블로그 코드 - https://yceffort.kr/2019/01/30/pytorch-3-convolutional-neural-network(2)
data = np.array(data, dtype="float32")
label = np.array(label, dtype="int64")


train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data, label, test_size=0.1)


train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()


test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()


train = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train, batch_size=32, shuffle=True)


# 신경망 구성
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 합성곱층
        self.conv1 = nn.Conv2d(1, 10, 5) # 입력 채널 수, 출력 채널 수, 필터 크기
        self.conv2 = nn.Conv2d(10, 20, 5)


        # 전결합층
        self.fc1 = nn.Linear(20 * 29 * 29, 50) # 29=(((((128-5)+1)/2)-5)+1)/2
        self.fc2 = nn.Linear(50, 9)
        #self.fc2 = nn.Linear(50, 2)


    def forward(self, x):
        # 풀링층
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) # 풀링 영역 크기
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 20 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
        #return x


# 인스턴스 생성
model = Net()
criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(500):
    total_loss = 0
    for train_x, train_y in train_loader:
        train_x, train_y = Variable(train_x), Variable(train_y)
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()
    if (epoch+1) % 50 == 0:
        print(epoch+1, total_loss)


test_x, test_y = Variable(test_X), Variable(test_Y)
result = torch.max(model(test_x).data, 1)[1]
accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())
print(accuracy)

tm = time.localtime(time.time())
print("End at: " + time.strftime("%c", tm))
