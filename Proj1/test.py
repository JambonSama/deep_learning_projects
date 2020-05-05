#!/usr/bin/env python
# ne pas enlever la ligne 1, ne pas mettre de commentaire au dessus

import torch
import time
from torch import Tensor
import dlc_practical_prologue as prologue

train_input, train_target, train_classes, test_input, test_target, test_classes  = prologue.generate_pair_sets(1000)

print("Done")