# -*- coding: utf-8 -*-

######################################################################

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

######################################################################


class AutoNet(nn.Module):
    """
    AutoNet class : Mother class from which our instanciated networks 
    inherit, this abstract class provides methods for automatic 
    network processing and benchmarking.
    """

    def __init__(self):
        super().__init__()

    def get_loss_number(self):
        """
        Pure virtual method which in daughter classes returns the number 
        of losses (for storing the losses in lists) returned by the 
        forward pass (1 without AL, 4 if AL).
        """
        raise NotImplementedError

    def measure_epoch_training_duration(self, train_loader):
        """
        Measures the time spent training on one epoch of train_loader.
        Parameter :
            train_loader -- DataLoader for training 
        """
        start = time.time()
        self.train_epoch(train_loader)
        end = time.time()
        return (end - start)

    def train_batch(self, parameter_list):
        """
        Pure virtual method which in daughter classes trains one batch.
        Parameter :
            parameter_list -- in daughter classes, the batch to train
        """
        raise NotImplementedError

    def train_epoch(self, train_loader):
        """
        Trains the network for one epoch on the DataLoader passed.
        Parameter :
            train_loader -- DataLoader on which to train
        """
        loss_array = []
        for batch in train_loader:
            loss = self.train_batch(batch)
            loss_array.append(loss)
        return loss_array

    def train_net(self, epoch_num, train_loader):
        """
        Trains the networks on the DataLoader passed for a certain
        number of epochs.
        Parameters :
            epoch_num -- number of epochs to train
            train_loader -- DataLoader on which to train
        """
        loss_array = []
        for i in range(epoch_num):
            losses = self.train_epoch(train_loader)
            loss_array.append(losses)
        return loss_array

    def test_net(self, parameter_list):
        """
        Pure virtual methods which in daughter classes test the network
        on the DataLoader passed and returns the accuracy on that set.
        Parameter : 
            test_loader -- DataLoader on which to test the network
        """
        raise NotImplementedError

    def determine_epoch_num(self, epoch_loader, valid_loader, max_epoch_num=200):
        """
        Estimates the optimal number of epochs for which to train the 
        network in order to avoid both over and under fitting.
        Parameters :
            epoch_loader -- DataLoader used to train the network
            valid_loader -- DataLoader used to verify wheter the training starts to overfit
            max_epoch_num -- max number of epoch on which to train
        """
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

######################################################################


class DigitNet(AutoNet):  # TODO: no separation
    """
    DigitNet class : This class is a network that takes a single 
    channel 14x14 grayscale image of digit and returns the digit class.
    """

    def __init__(self):
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
        # training variables
        self.crit = nn.CrossEntropyLoss()
        self.opti = optim.Adam(self.parameters())

    def forward(self, x):
        """
        Applies the forward pass.
        """
        x = torch.unsqueeze(x[:, 0, :, :], 1)  # unsqueeze for convl
        x = self.pool1(F.celu(self.conv1(x)))
        x = self.pool2(F.celu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.celu(self.fc1(x))
        x = F.celu(self.fc2(x))
        x = F.celu(self.fc3(x))
        return x

    def train_batch(self, batch):
        inputs, _, classes = batch
        self.opti.zero_grad()
        outputs = self(inputs)
        loss = self.crit(outputs, classes[:, 0])
        loss.backward()
        self.opti.step()
        return loss.item()

    def test_net(self, test_loader):
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, _, classes = data
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += classes[:, 0].size(0)
                correct += (predicted == classes[:, 0]).sum().item()
        return correct/total

    def get_loss_number(self):
        return 1

######################################################################


class NaiveNet(AutoNet):  # TODO: no separation
    """
    NaiveNet class : This class is a network that takes two images of 
    a digit in input, each on a single channel 14x14 tensor, and 
    outputs whether the first digit is lesser or equal to the second 
    (two classes). This net doesn't present either weight sharing nor 
    auxiliary losses.
    """

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
        """
        Applies the forward pass.
        """
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

    def get_loss_number(self):
        return 1

######################################################################


class WeightSharingNet(AutoNet):
    """
    WeightSharingNet class : This class is a network that takes two 
    images of a digit in input, each on a single channel 14x14 tensor, 
    and outputs whether the first digit is lesser or equal to the 
    second (two classes). This net presents weight sharing, but not 
    auxiliary losses.
    """

    def __init__(self):
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
        self.fc4 = nn.Linear(20, 2)
        # training variables
        self.crit = nn.CrossEntropyLoss()
        self.opti = optim.Adam(self.parameters())

    def forward(self, x):
        """
        Applies the forward pass.
        """
        x0 = torch.unsqueeze(x[:, 0, :, :], 1)  # unsqueeze for convl
        x0 = self.pool1(F.celu(self.conv1(x0)))
        x0 = self.pool2(F.celu(self.conv2(x0)))
        x0 = x0.view(-1, 16*5*5)
        x0 = F.celu(self.fc1(x0))
        x0 = F.celu(self.fc2(x0))
        x0 = F.celu(self.fc3(x0))

        x1 = torch.unsqueeze(x[:, 1, :, :], 1)
        x1 = self.pool1(F.celu(self.conv1(x1)))
        x1 = self.pool2(F.celu(self.conv2(x1)))
        x1 = x1.view(-1, 16*5*5)
        x1 = F.celu(self.fc1(x1))
        x1 = F.celu(self.fc2(x1))
        x1 = F.celu(self.fc3(x1))

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

    def get_loss_number(self):
        return 1

######################################################################


class AuxiliaryLossesNet(AutoNet):  # TODO: no separation
    """
    WeightSharingNet class : This class is a network that takes two 
    images of a digit in input, each on a single channel 14x14 tensor, 
    and outputs whether the first digit is lesser or equal to the 
    second (two classes). This net presents auxiliary losses, but not 
    weight sharing.
    """

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
        """
        Applies the forward pass.
        """
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

        x = torch.cat((x0, x1), 1)
        x = self.fc4(x)
        return x, x0, x1

    def train_batch(self, batch):
        inputs, targets, classes = batch
        self.opti.zero_grad()

        outputs, outputs0, outputs1 = self(inputs)

        loss = self.crit(outputs, targets)
        loss0 = self.crit(outputs0, classes[:, 0])
        loss1 = self.crit(outputs1, classes[:, 1])

        loss_total = 4*loss+loss0+loss1

        loss_total.backward()
        self.opti.step()
        return [loss_total.item(), loss.item(), loss0.item(), loss1.item()]

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

    def get_loss_number(self):
        return 4

######################################################################


class WsalNet(AutoNet):
    """
    WeightSharingNet class : This class is a network that takes two 
    images of a digit in input, each on a single channel 14x14 tensor, 
    and outputs whether the first digit is lesser or equal to the 
    second (two classes). This net presents both weight sharing and 
    auxiliary losses.
    """

    def __init__(self):
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
        self.fc4 = nn.Linear(20, 2)
        # training variables
        self.crit = nn.CrossEntropyLoss()
        self.opti = optim.Adam(self.parameters())

    def forward(self, x):
        """
        Applies the forward pass.
        """
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

        x = torch.cat((x0, x1), 1)
        x = self.fc4(x)
        return x, x0, x1

    def train_batch(self, batch):
        inputs, targets, classes = batch
        self.opti.zero_grad()

        outputs, outputs0, outputs1 = self(inputs)

        loss = self.crit(outputs, targets)
        loss0 = self.crit(outputs0, classes[:, 0])
        loss1 = self.crit(outputs1, classes[:, 1])

        loss_total = 4*loss+loss0+loss1

        loss_total.backward()
        self.opti.step()
        return [loss_total.item(), loss.item(), loss0.item(), loss1.item()]

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

    def get_loss_number(self):
        return 4
