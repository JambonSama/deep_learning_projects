
#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
# imports

import pickle
import torch
import networks as nw
import data_loader as dl
import results_analysis as ra

######################################################################
# params

N = 1000
BATCH_SIZE = 4
VALIDATION_RATIO = 0.2
ROUND_NUM = 3
NET_NUM = 2

if N//BATCH_SIZE != N/BATCH_SIZE:
    raise ArithmeticError("Batch number is not integer!")

######################################################################
# generating set loaders

valid_loader, epoch_loader, train_loader, test_loader = dl.generate_set_loaders(
    N, BATCH_SIZE, VALIDATION_RATIO)

######################################################################
# array inits for results analysis

epoch_training_duration = torch.zeros([NET_NUM])
param_nums = torch.zeros([NET_NUM])
epoch_nums = torch.zeros([ROUND_NUM, NET_NUM])
loss_arrays = []
accuracies = torch.zeros([ROUND_NUM, NET_NUM])

#####################################################################
# determining a "good" number of epochs for each net

# looping for statistical soundness
for i in range(ROUND_NUM):
    # nets inits
    nets = [nw.DigitNet(), nw.NaiveNet(), nw.WeightSharingNet(),
            nw.AuxiliaryLossesNet(), nw.WsalNet()]
    for j, net in enumerate(nets):
        # determining "good" epoch number
        epoch_num = net.determine_epoch_num(epoch_loader, valid_loader)
        epoch_nums[i][j] = epoch_num

# averaging
epoch_nums = torch.round(torch.mean(epoch_nums, 0)).int()

######################################################################
# training and testing all the nets

# looping for statistical soundness
for i in range(ROUND_NUM):
    # nets inits
    nets = [nw.DigitNet(), nw.NaiveNet(), nw.WeightSharingNet(),
            nw.AuxiliaryLossesNet(), nw.WsalNet()]

    losses = []
    for j, net in enumerate(nets):
        # net training
        loss = net.train_net(epoch_nums[j], train_loader)
        losses.append(loss)

        # nets testing
        accuracy = net.test_net(test_loader)
        accuracies[i, j] = accuracy

    # results archiving
    loss_arrays.append(losses)

######################################################################
# benchmarking nets

nets = [nw.DigitNet(), nw.NaiveNet(), nw.WeightSharingNet(),
        nw.AuxiliaryLossesNet(), nw.WsalNet()]
for i, net in enumerate(nets, 0):
    epoch_training_duration[i] = net.measure_epoch_training_duration(
        train_loader)
    param_nums[i] = sum(p.numel() for p in net.parameters() if p.requires_grad)

######################################################################
# results archiving

f = open('results.pck', 'wb')
pickle.dump((epoch_training_duration, param_nums,
             epoch_nums, loss_arrays, accuracies), f)

######################################################################
# results displaying

ra.prepare_results(epoch_training_duration, param_nums,
                   epoch_nums, loss_arrays, accuracies)

######################################################################
# end of script

print("Done")
