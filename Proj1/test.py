#!/usr/bin/env python
# ne pas enlever la ligne 1, ne pas mettre de commentaire au dessus

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

import time

import numpy as np

import dlc_practical_prologue as prologue

train_input, train_target, train_classes, test_input, test_target, test_classes  = prologue.generate_pair_sets(1000)

train_set = torch.utils.data.TensorDataset(train_input, train_target)
test_set = torch.utils.data.TensorDataset(test_input, test_target)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
targets = ("0", "1")

# Doc : 
# http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf (LeNet-5)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # conv layers
        self.conv = nn.Conv2d(2, 16, 5)
        # pooling layers
        self.pool = nn.MaxPool2d(2)
        # fully connected layers
        self.fc1 = nn.Linear(16*5*5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 16*5*5) # reshape for fcl
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Doc : 
# ee559-handout-4-1-DAG-networks.pdf
# https://pytorch.org/tutorials/beginner/examples_nn/dynamic_net.html
class WeightSharingNet(nn.Module):
    def __init__(self):
        super().__init__()
        # conv layers
        self.conv = nn.Conv2d(1, 16, 5)
        # pooling layers
        self.pool = nn.MaxPool2d(2)
        # fully connected layers
        self.fc1 = nn.Linear(16*5*5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x0 = self.pool(F.relu(self.conv(torch.unsqueeze(x[:,0,:,:],1))))
        x0 = x0.view(-1, 16*5*5) # reshape for fcl
        x0 = F.relu(self.fc1(x0))
        x0 = F.relu(self.fc2(x0))
        x0 = F.relu(self.fc3(x0))

        x1 = self.pool(F.relu(self.conv(torch.unsqueeze(x[:,1,:,:],1))))
        x1 = x1.view(-1, 16*5*5) # reshape for fcl
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))

        x = self.fc4(x0-x1)
        return x

class AuxiliaryLossesNet(nn.Module):
    def __init__(self):
        super().__init__()

net1 = Net()
net2 = WeightSharingNet()

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(net1.parameters())
optimizer2 = optim.Adam(net2.parameters())

a = 20
epoch_num = 25
losses = np.zeros((3,epoch_num*250))

for epoch in range(epoch_num):  # loop over the dataset multiple times
    running_loss1 = 0.0
    running_loss2 = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, targets]
        inputs, targets = data

        # zero the parameter gradients
        optimizer1.zero_grad() # TO UNDERSTAND see handout 8-5 p5
        optimizer2.zero_grad()
        
        # forward + backward + optimize
        outputs1 = net1(inputs)
        outputs2 = net2(inputs)
        loss1 = criterion(outputs1, targets)
        loss2 = criterion(outputs2, targets)
        loss1.backward()
        loss2.backward()
        optimizer1.step()
        optimizer2.step()

        # print statistics
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
        if i % a == a-1:    # print every a mini-batches
            print(f"[{epoch+1:2d} {i+1:4d}] loss={running_loss1/a:.6f}")
            print(f"[{epoch+1:2d} {i+1:4d}] loss={running_loss2/a:.6f}")
            running_loss1 = 0.0
            running_loss2 = 0.0

print('Finished Training')

correct1 = 0
correct2 = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        inputs, targets = data

        outputs1 = net1(inputs)
        outputs2 = net2(inputs)
        
        _, predicted1 = torch.max(outputs1.data,1)
        _, predicted2 = torch.max(outputs2.data,1)

        total += targets.size(0)
        correct1 += (predicted1==targets).sum().item()
        correct2 += (predicted2==targets).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct1 / total} %")
print(f"Accuracy of the network on the 10000 test images: {100 * correct2 / total} %")
