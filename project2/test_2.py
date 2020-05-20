#!/usr/bin/env python
# ne pas enlever la ligne 1, ne pas mettre de commentaire au dessus

# Import
import numpy as np
import torch
import neural_network as nn
import layer
torch.set_grad_enabled(False)
       
def main():
    # Parameters for train and test set generation
    print("**** Train and test set generation ****")
    torch.random.manual_seed(42)
    center = [0.5,0.5]
    radius2 = 1.0/(2*np.pi)

    # Training set
    train_set = torch.rand(1000,2)
    train_distance = ((train_set[:,0]-center[0]).pow(2)+(train_set[:,1]-center[1]).pow(2))
    train_label = torch.stack((torch.tensor([1 if i < radius2 else 0 for i in train_distance]),torch.tensor([1 if i >= radius2 else 0 for i in train_distance])),dim=1)

    # Test set
    test_set = torch.rand(1000,2)
    test_distance = ((test_set[:,0]-center[0]).pow(2)+(test_set[:,1]-center[1]).pow(2))
    test_label = torch.tensor([1 if i < radius2 else 0 for i in test_distance])

    # Neural network
    print("**** Creation of the neural network with 3 hidden layers, all ReLu ****")
    net = nn.NeuralNetwork()
    net.sequential(layer.ReLu(2,25), layer.ReLu(25,25), layer.ReLu(25,25), layer.ReLu(25,2))
    print("**** Training the neural network ****")
    net.train_network(train_set, train_label, epochs=50, batch_size=1, learning_rate=0.02, print_error=True, test_set=test_set, test_label=test_label)
    print("**** Training done ****")
    net.params()

if __name__ == "__main__":
    main()
    print("Done")   