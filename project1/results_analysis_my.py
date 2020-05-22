# -*- coding: utf-8 -*-

######################################################################

import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np  # only for plots + csv export

######################################################################


def prepare_results(epoch_training_durations, param_nums, epoch_nums, loss_arrays, accuracies):
    dirName = "results"
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    
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
    np.savetxt(format(f"{dirName}/net_duration_res.csv"), epoch_training_durations.numpy())
    np.savetxt(format(f"{dirName}/net_param_nums_res.csv"), param_nums.numpy())
    np.savetxt(format(f"{dirName}/net_epochs_res.csv"), epoch_nums.numpy())
    np.savetxt(format(f"{dirName}/net_acc_res_mean.csv"), accuracies_mean.numpy())
    np.savetxt(format(f"{dirName}/net_acc_res_std.csv"), accuracies_std.numpy())

    # figures
    fig0, ax0 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for i in range(1,len(loss_arrays_mean)):
        # save data
        np.savetxt(format(f"net{i}_loss_res_mean.csv"),
                   loss_arrays_mean[0].numpy())
        np.savetxt(format(f"net{i}_loss_res_std.csv"),
                   loss_arrays_std[0].numpy())

        # plots
        x = range(1, 1+loss_arrays_mean[i].shape[0])
        if len(loss_arrays_mean[i].shape) == 2:
            ax0.errorbar(x, loss_arrays_mean[i][:, 0].numpy(
            ), loss_arrays_std[i][:, 0].numpy())
            fig1, ax1 = plt.subplots()
            for j in range(loss_arrays_mean[i].shape[1]):
                ax1.errorbar(x, loss_arrays_mean[i][:, j].numpy(
                ), loss_arrays_std[i][:, j].numpy())
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Loss")
            ax1.legend(["Total", "Target", "Class 0", "Class 1"])
            fig1.savefig(f"{dirName}/fig1_{i}.svg")

        elif len(loss_arrays_mean[i].shape) == 1:
            ax0.errorbar(
                x, loss_arrays_mean[i].numpy(), loss_arrays_std[i].numpy())

    ax2.errorbar(range(1, 1+loss_arrays_mean[0].shape[0]), loss_arrays_mean[0].numpy(
            ), loss_arrays_std[0].numpy())

    ax0.set_xlabel("Epochs")
    ax0.set_ylabel("Loss")
    ax0.legend(["naive net", "WS net", "AL net", "WS+AL net"])
    fig0.savefig(f"{dirName}/fig0.svg")

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    fig2.savefig(f"{dirName}/fig2.svg")

    # plt.show()

    print(f"\t\t\t\tDCM \t\tnaive net \tWS net \t\tAL net \t\tWS+AL net")
    print("Accuracy (mean) \t\t", end=" ")
    for a in accuracies_mean:
        print(f"{a*100:2.1f}\t\t", end=" ")
    print("")
    print("Accuracy (std) \t\t\t", end=" ")
    for a in accuracies_std:
        print(f"{a*100:2.1f}\t\t", end=" ")
    print("")
    print("Number of epoch \t\t", end=" ")
    for a in epoch_nums:
        print(f"{a}\t\t", end=" ")
    print("")
    print("Epoch training duration [s] \t", end=" ")
    for a in epoch_training_durations:
        print(f"{a:2.3f}\t\t", end=" ")
    print("")
    print("Number of parameters \t\t", end=" ")
    for a in param_nums:
        print(f"{a}\t", end=" ")
    print("")
 
    return

######################################################################


f = open("results.pck", "rb")
data = pickle.load(f)
d = data[0]
p = data[1]
e = data[2]
l = data[3]
a = data[4]

prepare_results(d, p, e, l, a)

print("Done")
