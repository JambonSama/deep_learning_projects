# -*- coding: utf-8 -*-

######################################################################

import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np  # only for plots + csv export

######################################################################


def prepare_results(epoch_training_duration, param_nums, epoch_nums, loss_arrays, accuracies):
    # accuracy
    accuracies_mean = torch.mean(accuracies, 0)
    accuracies_std = torch.std(accuracies, 0)

    # loss (averaged over epochs)
    loss_arrays = [torch.tensor([loss_arrays[i][net_index] for i in range(
        len(loss_arrays))]) for net_index in range(len(loss_arrays[0]))]
    loss_arrays = [torch.mean(loss_arr, 2) for loss_arr in loss_arrays]
    loss_arrays_mean = [torch.mean(loss_arr, 0) for loss_arr in loss_arrays]
    loss_arrays_std = [torch.std(loss_arr, 0) for loss_arr in loss_arrays]

    # save data
    np.savetxt(format('net_duration_res_mean.csv'),
               epoch_training_duration.numpy())
    np.savetxt(format('net_param_nums_res_std.csv'), param_nums.numpy())
    np.savetxt(format('net_epochs_res_mean.csv'), epoch_nums.numpy())
    np.savetxt(format('net_acc_res_mean.csv'), accuracies_mean.numpy())
    np.savetxt(format('net_acc_res_std.csv'), accuracies_std.numpy())

    # figures
    fig0, ax0 = plt.subplots()

    for i in range(len(loss_arrays_mean)):
        # save data
        np.savetxt(format(f'net{i}_loss_res_mean.csv'),
                   loss_arrays_mean[0].numpy())
        np.savetxt(format(f'net{i}_loss_res_std.csv'),
                   loss_arrays_std[0].numpy())

        # plots
        x = range(1, 1+loss_arrays_mean[i].shape[0])
        if len(loss_arrays_mean[i].shape) == 2:
            ax0.errorbar(x=range(1, 1+loss_arrays_mean[i].shape[0]), y=loss_arrays_mean[i][:, 0].numpy(
            ), yerr=loss_arrays_std[i][:, 0].numpy())
            fig1, ax1 = plt.subplots()
            for j in range(loss_arrays_mean[i].shape[1]):
                ax1.errorbar(x, loss_arrays_mean[i][:, j].numpy(
                ), loss_arrays_std[i][:, j].numpy())
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend(['Total', 'Target', 'Class 0', 'Class 1'])
            fig1.savefig(f'fig1_{i}.svg')

        elif len(loss_arrays_mean[i].shape) == 1:
            ax0.errorbar(
                x, loss_arrays_mean[i].numpy(), loss_arrays_std[i].numpy())

    ax0.set_xlabel('Epochs')
    ax0.set_ylabel('Loss')
    ax0.legend(['DCM', 'naive net', 'WS net', 'AL net', 'WS+AL net'])
    fig0.savefig('fig0.svg')
    plt.show()

    return

######################################################################


# f = open('results.pck', 'rb')
# data = pickle.load(f)
# e = data[0]
# l = data[1]
# a = data[2]

# prepare_results(e, l, a)

# print("Done")
