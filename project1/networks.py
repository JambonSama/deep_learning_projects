# -*- coding: utf-8 -*-

#########
# IMPORTS
#########

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


######################
# NETS IMPLEMENTATIONS
######################


class AutoNet(nn.Module):
    def __init__(self):
        super().__init__()

    def train_batch(self, parameter_list):
        raise NotImplementedError

    def train_epoch(self, train_loader, loss_array=None, epoch_index=None):
        for batch_index, batch in enumerate(train_loader, 0):
            loss = self.train_batch(batch)
            if((loss_array is not None) and (epoch_index is not None)):
                loss_array[epoch_index, batch_index] = loss
        return

    def train_net(self, epoch_num, train_loader, loss_array):
        for epoch_index in range(epoch_num):
            self.train_epoch(train_loader, loss_array, epoch_index)
        return

    def test_net(self, parameter_list):
        raise NotImplementedError

    def determine_epoch_num(self, epoch_loader, valid_loader, max_epoch_num=200):
        perf = self.test_net(valid_loader)
        threshold = 20
        cummulative_overfit = 0
        for epoch_index in range(max_epoch_num):
            self.train_epoch(epoch_loader)
            new_perf = self.test_net(valid_loader)
            if new_perf < perf:
                cummulative_overfit += 1
                if cummulative_overfit > threshold:
                    break
            else:
                perf = new_perf
                cummulative_overfit = 0
        return epoch_index - threshold

# Doc:
# http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf (LeNet-5)


class Net(AutoNet):
    def __init__(self):
        super().__init__()
        # convolutional layers
        self.conv1_0 = nn.Conv2d(1, 6, 3)
        self.conv2_0 = nn.Conv2d(6, 16, 3)
        self.conv1_1 = nn.Conv2d(1, 6, 3)
        self.conv2_1 = nn.Conv2d(6, 16, 3)
        # pooling layers
        self.pool1 = nn.MaxPool2d(2, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        # fully connected layers
        self.fc1_0 = nn.Linear(16*5*5, 120)
        self.fc2_0 = nn.Linear(120, 84)
        self.fc3_0 = nn.Linear(84, 10)
        self.fc1_1 = nn.Linear(16*5*5, 120)
        self.fc2_1 = nn.Linear(120, 84)
        self.fc3_1 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(20, 2)
        # training variables
        self.crit = nn.CrossEntropyLoss()
        self.opti = optim.Adam(self.parameters())

    def forward(self, x):
        x0 = torch.unsqueeze(x[:, 0, :, :], 1)  # unsqueeze for convl
        x0 = self.pool1(F.celu(self.conv1_0(x0)))
        x0 = self.pool2(F.celu(self.conv2_0(x0)))
        x0 = x0.view(-1, 16*5*5)
        x0 = F.celu(self.fc1_0(x0))
        x0 = F.celu(self.fc2_0(x0))
        x0 = F.celu(self.fc3_0(x0))

        x1 = torch.unsqueeze(x[:, 1, :, :], 1)
        x1 = self.pool1(F.celu(self.conv1_1(x1)))
        x1 = self.pool2(F.celu(self.conv2_1(x1)))
        x1 = x1.view(-1, 16*5*5)
        x1 = F.celu(self.fc1_1(x1))
        x1 = F.celu(self.fc2_1(x1))
        x1 = F.celu(self.fc3_1(x1))

        x = torch.cat((x0, x1), 1)
        x = self.fc4(x)
        return x

    def train_batch(self, batch):
        inputs, targets, _ = batch
        self.opti.zero_grad()
        outputs = self(inputs)
        loss = self.crit(outputs, targets)
        loss.backward()
        self.opti.step()
        return loss.item()

    def test_net(self, test_loader):
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets, _ = data
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        return correct/total

# Doc:
# ee559-handout-4-1-DAG-networks.pdf
# https://pytorch.org/tutorials/beginner/examples_nn/dynamic_net.html


class WeightSharingNet(AutoNet):
    def __init__(self, auxiliary_flag=False):
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # pooling layers
        self.pool1 = nn.MaxPool2d(2, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        # fully connected layers
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)
        # auxiliary ?
        self.auxiliary_flag = auxiliary_flag
        # training variables
        self.crit = nn.CrossEntropyLoss()
        self.opti = optim.Adam(self.parameters())

    def forward(self, x):
        x0 = torch.unsqueeze(x[:, 0, :, :], 1)  # unsqueeze for convl
        x0 = self.pool1(F.celu(self.conv1(x0)))
        x0 = self.pool2(F.celu(self.conv2(x0)))
        x0 = x0.view(-1, 16*5*5)
        x0 = F.celu(self.fc1(x0))
        x0 = F.celu(self.fc2(x0))
        x0 = self.fc3(x0)

        x1 = torch.unsqueeze(x[:, 1, :, :], 1)
        x1 = self.pool1(F.celu(self.conv1(x1)))
        x1 = self.pool2(F.celu(self.conv2(x1)))
        x1 = x1.view(-1, 16*5*5)
        x1 = F.celu(self.fc1(x1))
        x1 = F.celu(self.fc2(x1))
        x1 = self.fc3(x1)

        x = self.fc4(F.celu(x0-x1))
        return x, x0, x1

    def train_batch(self, batch):
        inputs, targets, classes = batch
        self.opti.zero_grad()

        outputs, outputs0, outputs1 = self(inputs)

        loss = self.crit(outputs, targets)
        loss0 = self.crit(outputs0, classes[:, 0])
        loss1 = self.crit(outputs1, classes[:, 1])

        if(self.auxiliary_flag):
            loss = 4*loss+loss0+loss1

        loss.backward()
        self.opti.step()
        return loss.item()

    def test_net(self, test_loader):
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets, _ = data
                outputs, _, _ = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        return correct/total

# Doc:
# https://stats.stackexchange.com/questions/304699/what-is-auxiliary-loss-as-mentioned-in-pspnet-paper/


class AuxiliaryLossesNet(AutoNet):
    def __init__(self):
        super().__init__()
        # convolutional layers
        self.conv1_0 = nn.Conv2d(1, 6, 3)
        self.conv2_0 = nn.Conv2d(6, 16, 3)
        self.conv1_1 = nn.Conv2d(1, 6, 3)
        self.conv2_1 = nn.Conv2d(6, 16, 3)
        # pooling layers
        self.pool1 = nn.MaxPool2d(2, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        # fully connected layers
        self.fc1_0 = nn.Linear(16*5*5, 120)
        self.fc2_0 = nn.Linear(120, 84)
        self.fc3_0 = nn.Linear(84, 10)
        self.fc1_1 = nn.Linear(16*5*5, 120)
        self.fc2_1 = nn.Linear(120, 84)
        self.fc3_1 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)
        # training variables
        self.crit = nn.CrossEntropyLoss()
        self.opti = optim.Adam(self.parameters())

    def forward(self, x):
        x0 = torch.unsqueeze(x[:, 0, :, :], 1)  # unsqueeze for convl
        x0 = self.pool1(F.celu(self.conv1_0(x0)))
        x0 = self.pool2(F.celu(self.conv2_0(x0)))
        x0 = x0.view(-1, 16*5*5)
        x0 = F.celu(self.fc1_0(x0))
        x0 = F.celu(self.fc2_0(x0))
        x0 = self.fc3_0(x0)

        x1 = torch.unsqueeze(x[:, 1, :, :], 1)
        x1 = self.pool1(F.celu(self.conv1_1(x1)))
        x1 = self.pool2(F.celu(self.conv2_1(x1)))
        x1 = x1.view(-1, 16*5*5)
        x1 = F.celu(self.fc1_1(x1))
        x1 = F.celu(self.fc2_1(x1))
        x1 = self.fc3_1(x1)

        x = self.fc4(F.celu(x0-x1))
        return x, x0, x1

    def train_batch(self, batch):
        inputs, targets, classes = batch
        self.opti.zero_grad()

        outputs, outputs0, outputs1 = self(inputs)

        loss = self.crit(outputs, targets)
        loss0 = self.crit(outputs0, classes[:, 0])
        loss1 = self.crit(outputs1, classes[:, 1])

        loss = 4*loss+loss0+loss1

        loss.backward()
        self.opti.step()
        return loss.item()

    def test_net(self, test_loader):
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets, _ = data
                outputs, _, _ = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        return correct/total
