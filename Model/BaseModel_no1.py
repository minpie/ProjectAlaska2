import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection
from tqdm import tqdm

meta_info_path = '/storages/hdd0/pt0/ProjectAlaska2Datas/Datasets/track_a_learn_label.csv'
data_path = "/storages/hdd0/pt0/ProjectAlaska2Datas/Datasets/output_images/StreamOrder/"

df = pd.read_csv(meta_info_path)
df['filename'] = df['filename'].apply(str)

dirs = df['type'].drop_duplicates().tolist()
data, label = [], []

for i, d in enumerate(dirs):
    print(i, 'and', d)

files = os.listdir(data_path)
for f in files[:10000]:
    #이미지 오픈
    img = Image.open(data_path + "/" + f)
    f = f.rstrip('.png')

    #이미지 라벨 찾기
    find_label = df['type'].loc[df['filename'] == f].to_string(index=False)
    # 이미지를 128,128로 리사이즈
    resize_img = img.resize((128,128))
    resize_img = np.array(resize_img)
    data.append(resize_img)

    # 라벨
    dirs_num = list(enumerate(dirs))
    dirs_num = dict(dirs_num)
    i = [k for k, v in dirs_num.items() if v == find_label][0]
    # 라벨 (0, hIVJ    1, EQbM    2, BghB    3, PRoU    4, nqBD    5, wbxA    6, KOPU    7, pvpk    8, aMyf)
    label.append(i)

# 블로그 참고 - https://yceffort.kr/2019/01/30/pytorch-3-convolutional-neural-network(2)
data = np.array(data, dtype='float32')
label = np.array(label, dtype='int64')




train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data, label, test_size=0.1)

train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()

test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()

train = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train, batch_size=1, shuffle=True)

test = TensorDataset(test_X, test_Y)
test_loader = DataLoader(test, batch_size=1, shuffle=True)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 합성곱층
        self.conv1 = nn.Conv2d(1, 10, 5) # 입력 채널 수, 출력 채널 수, 필터 크기
        self.conv2 = nn.Conv2d(10, 20, 5)

        # 전결합층
        self.fc1 = nn.Linear(20 * 29 * 29, 50) # 29=(((((128-5)+1)/2)-5)+1)/2
        self.fc2 = nn.Linear(50, 9)

    def forward(self, x):
        # 풀링층
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) # 풀링 영역 크기
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 20 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

# 인스턴스 생성
model = Net()
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in tqdm(range(30)):
  total_loss = 0
  for train_x, train_y in train_loader:
    train_x, train_y = Variable(train_x), Variable(train_y)
    optimizer.zero_grad()
    output = model(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()
    total_loss += loss.data.item()
  print("\n", epoch, loss.data.item())

# Function to test the model
def test():
    # Load the model that we saved at the end of the training loop
    model = Net()

    running_accuracy = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            outputs = outputs.to(torch.float32)
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()
        print('inputs is: %d %%' % (100 * running_accuracy / total))

test()
