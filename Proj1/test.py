
#!/usr/bin/env python
# -*- coding: utf-8 -*-


#########
# IMPORTS
#########

import numpy as np
import networks as networks
import data as data


# params
N = 1000
BATCH_SIZE = 4
VALIDATION_RATIO = 0.2
batch_num = N//BATCH_SIZE

# set loaders
valid_loader, epoch_loader, train_loader, test_loader = data.generate_set_loaders(
    N, BATCH_SIZE, VALIDATION_RATIO)

# init net
net = networks.Net()

# determine optimal epoch number
epoch_num = net.determine_epoch_num(epoch_loader, valid_loader)

# init loss array for results
loss_array = np.ndarray((epoch_num, batch_num))

# train net
net.train_net(epoch_num, train_loader, loss_array)

# test net
performance = net.test_net(test_loader)

# print results
print(f"Performance : {performance}")