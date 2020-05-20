# -*- coding: utf-8 -*-

import torch
import numpy as np
import dlc_practical_prologue as prologue

def generate_set_loaders(N=1000, batch_size=4, percentage=0.2):
    # data loading
    train_input, train_target, train_classes, test_input, test_target, test_classes  = prologue.generate_pair_sets(N)

    # sets
    indices = np.random.permutation(range(N))
    n = round(N*percentage)

    valid_set = torch.utils.data.TensorDataset(train_input[indices[:n]], train_target[indices[:n]], train_classes[indices[:n]])
    epoch_set = torch.utils.data.TensorDataset(train_input[indices[n:]], train_target[indices[n:]], train_classes[indices[n:]])
    train_set = torch.utils.data.TensorDataset(train_input, train_target, train_classes)
    test__set = torch.utils.data.TensorDataset(test_input, test_target, test_classes)

    # batches
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size, shuffle=True, num_workers=2)
    epoch_loader = torch.utils.data.DataLoader(epoch_set, batch_size, shuffle=True, num_workers=2)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=2)
    test__loader = torch.utils.data.DataLoader(test__set, batch_size, shuffle=False, num_workers=2)

    return valid_loader, epoch_loader, train_loader, test__loader