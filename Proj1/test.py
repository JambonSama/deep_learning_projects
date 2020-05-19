#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########
# IMPORTS
#########
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

import time

import numpy as np

import matplotlib.pyplot as plt

import dlc_practical_prologue as prologue

############################
# data loading and formating
############################
N = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes  = prologue.generate_pair_sets(N)

train_set = torch.utils.data.TensorDataset(train_input, train_target)
train_set2 = torch.utils.data.TensorDataset(train_input, train_target, train_classes)
test_set = torch.utils.data.TensorDataset(test_input, test_target)
test_set2 = torch.utils.data.TensorDataset(test_input, test_target, test_classes)

batch_size = 4
train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=2)
train_loader2 = torch.utils.data.DataLoader(train_set2, batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=2)
test_loader2 = torch.utils.data.DataLoader(test_set2, batch_size, shuffle=False, num_workers=2)

classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
targets = ("0", "1")

######################
# nets implementations
######################
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
        x0 = torch.unsqueeze(x[:,0,:,:],1)
        x0 = self.pool(F.relu(self.conv(x0)))
        x0 = x0.view(-1, 16*5*5) # reshape for fcl
        x0 = F.relu(self.fc1(x0))
        x0 = F.relu(self.fc2(x0))
        x0 = F.relu(self.fc3(x0))

        x1 = torch.unsqueeze(x[:,1,:,:],1)
        x1 = self.pool(F.relu(self.conv(x1)))
        x1 = x1.view(-1, 16*5*5) # reshape for fcl
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))

        x = self.fc4(x0-x1)
        return x

# Doc : 
# https://stats.stackexchange.com/questions/304699/what-is-auxiliary-loss-as-mentioned-in-pspnet-paper/
class AuxiliaryLossesNet(nn.Module):
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
        x0 = torch.unsqueeze(x[:,0,:,:],1)
        x0 = self.pool(F.relu(self.conv(x0)))
        x0 = x0.view(-1, 16*5*5) # reshape for fcl
        x0 = F.relu(self.fc1(x0))
        x0 = F.relu(self.fc2(x0))
        x0 = F.relu(self.fc3(x0))

        x1 = torch.unsqueeze(x[:,1,:,:],1)
        x1 = self.pool(F.relu(self.conv(x1)))
        x1 = x1.view(-1, 16*5*5) # reshape for fcl
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))

        x = self.fc4(x0-x1)
        return x,x0,x1

#####################
# nets instanciations
#####################
net0 = Net()
net1 = WeightSharingNet()
net2 = AuxiliaryLossesNet()

criterion = nn.CrossEntropyLoss()
optimizer0 = optim.Adam(net0.parameters())
optimizer1 = optim.Adam(net1.parameters())
optimizer2 = optim.Adam(net2.parameters())

##########
# training
##########
epoch_num = 25
epoch_losses = np.zeros((epoch_num,3))

batch_num = N//batch_size
batch_losses = np.zeros((epoch_num*batch_num,3))

bar_length = 50
batch_stride = 20

for epoch in range(epoch_num):  # loop over the dataset multiple times
    # progress printing
    overall_fraction = (epoch)/epoch_num
    overall_progress = int(overall_fraction*bar_length)
    overall_bar = '█'*overall_progress + '-'*(bar_length-overall_progress)

    for i, (_, data), (_, data2) in zip(range(batch_num), enumerate(train_loader, 0),enumerate(train_loader2, 0)):
        # get the inputs; data is a list of [inputs, targets]
        inputs, targets = data
        inputs2, targets2, classes2 = data2

        # zero the parameter gradients
        optimizer0.zero_grad() # TO UNDERSTAND see handout 8-5 p5
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        # forward + backward + optimize
        outputs0 = net0(inputs)
        loss0 = criterion(outputs0, targets)
        loss0.backward()
        optimizer0.step()

        outputs1 = net1(inputs)
        loss1 = criterion(outputs1, targets)
        loss1.backward()
        optimizer1.step()

        outputs2,op2_0,op2_1 = net2(inputs2)
        loss2_ = criterion(outputs2, targets2)
        loss2_0 = criterion(op2_0, classes2[:,0])
        loss2_1 = criterion(op2_1, classes2[:,1])
        loss2 = loss2_*0.6+(loss2_0+loss2_1)*0.4
        loss2.backward()
        optimizer2.step()

        # statistics
        batch_losses[epoch*batch_num+i,:] = np.array([loss0.item(), loss1.item(), loss2.item()])
        epoch_losses[epoch,:] += np.array([loss0.item(), loss1.item(), loss2.item()])

        # progress printing
        if i % batch_stride == batch_stride-1:    # print every a mini-batches
            epoch_fraction = (i)/batch_num
            epoch_progress = int(epoch_fraction*bar_length)
            epoch_bar = '█'*epoch_progress + '-'*(bar_length-epoch_progress)
            print(f"\rOverall training progress : |{overall_bar}| {overall_fraction*100:3.2f}% complete, current epoch training progress : |{epoch_bar}| {epoch_fraction*100:3.2f}% complete", end = "\r")

print(f"\rOverall training progress : |{'█'*bar_length}| {100:3.2f}% complete, current epoch training progress : |{'█'*bar_length}| {100:3.2f}% complete")
print("Training complete")

##############
# losses plots
##############
legend = ("Bare net","Weight sharing net","Auxiliary losses")
n_lin = 1
n_col = 2
size = 6
fig, ax = plt.subplots(n_lin, n_col, figsize=(n_col*size, n_lin*size))

ax[0].plot(batch_losses[:,0])
ax[0].plot(batch_losses[:,1])
ax[0].plot(batch_losses[:,2])
ax[0].set_title(f"Training loss plot (for each batch)")
ax[0].legend(legend)

ax[1].plot(epoch_losses[:,0])
ax[1].plot(epoch_losses[:,1])
ax[1].plot(epoch_losses[:,2])
ax[1].set_title(f"Training loss plot (for each epoch)")
ax[1].legend(legend)

plt.show()

#########
# testing
######### 
correct0 = 0
correct1 = 0
correct2 = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        inputs, targets = data

        outputs0 = net0(inputs)
        outputs1 = net1(inputs)
        outputs2,_,_ = net2(inputs)
        
        _, predicted0 = torch.max(outputs0.data,1)
        _, predicted1 = torch.max(outputs1.data,1)
        _, predicted2 = torch.max(outputs2.data,1)

        total += targets.size(0)
        correct0 += (predicted0==targets).sum().item()
        correct1 += (predicted1==targets).sum().item()
        correct2 += (predicted2==targets).sum().item()

#########
# results
#########
print(f"Accuracy of the network on the {N} test images: ")
print(f"\tBare net:\t\t{100 * correct0 / total:2.1f} %")
print(f"\tWeight sharing net: \t{100 * correct1 / total:2.1f} %")
print(f"\tAuxiliary losses net: \t{100 * correct2 / total:2.1f} %")
