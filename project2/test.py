#!/usr/bin/env python
# ne pas enlever la ligne 1, ne pas mettre de commentaire au dessus

# Import
import numpy as np
import torch
torch.set_grad_enabled(False)

# Train and test set generation

torch.random.manual_seed(42)
center = [0.5,0.5]

radius2 = 1.0/(2*np.pi)

# Training set
train_set = torch.rand(1000,2)
train_distance = ((train_set[:,0]-center[0]).pow(2)+(train_set[:,1]-center[1]).pow(2))
train_label = torch.stack(((train_distance<radius2).int(),(train_distance>=radius2).int()),dim=1)

# Test set
test_set = torch.rand(1000,2)
test_distance = ((test_set[:,0]-center[0]).pow(2)+(test_set[:,1]-center[1]).pow(2))
test_label = torch.stack(((test_distance<radius2).int(),(test_distance>=radius2).int()),dim=1)


def loss(x, target):
    out = torch.mean((target-x).pow(2),0)
    return out
    
def d_loss(output, target):
    out = 2.0 * (output - target)/len(output)
    return out
    
def test_error(test_input, test_target):
    #print(f"test_input{test_input}")
    #print(f"test_target{test_target}")
    temp = ((test_input == test_target).int()).sum(dim=1)
    out = ((temp != 2).int()).sum()
    return out


class Layer:
    
    def __init__(self, nb_input, nb_output):
        self.variance = 2.0 / (nb_input + nb_output)
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.weight = np.sqrt(self.variance)*torch.randn(nb_input, nb_output)
        self.bias = torch.zeros(nb_output)

    def foward_pass(self,x):
        #print("forward pass")
        self.input_matrix = x
        self.z = torch.matmul(x, self.weight)+self.bias
        out = self.activation(self.z)
        self.output_matrix = out
        return out
    
    def backward_pass(self, dl_dy):
        #print("backward pass")
        dl_dz = self.d_activation(self.z)
        dl_dx = torch.matmul(dl_dz*dl_dy,self.weight.T)
        dl_dw = torch.matmul(self.input_matrix.view(-1,1), (dl_dz*dl_dy).view(1,-1))
        dl_db = dl_dz*dl_dy
        #print(f"dl_dw {dl_dw}, dl_db {dl_db}")
        # Accumulation parameter
        self.acc_weight = self.acc_weight + dl_dw
        self.acc_bias = self.acc_bias + dl_db
        return dl_dx
    
    def update_parameters(self, learning_rate, batch_size):
        #print(f"acc_weight {self.acc_weight}, acc_bias {self.acc_bias}")
        self.weight = self.weight - learning_rate * self.acc_weight/batch_size
        self.bias = self.bias - learning_rate * self.acc_bias/batch_size
        
        
    def params(self):
        print(f"Nb input : {self.nb_input}")
        print(f"Nb output : {self.nb_output}")
        print(f"Weight matrix shape : {self.weight.shape}")
        print(f"Bias matrix shape : {self.bias.shape}")
        print(f"Weight: {self.weight}")
        print(f"Bias : {self.bias}")
        
class LinearLayer(Layer):
    def activation(self,x):
        out = x
        return out
    
    
class ReLuLayer(Layer):
    def activation(self,x):
        #print("ReLu activation function")
        out = (x>0).int()*x
        return out
    
    def d_activation(self, output):
        #print("ReLu derivate activation function")
        out = (output>0).int()
        return out
    
class TanHLayer(Layer):
    def activation(self,x):
        #print("TanH activation function")
        out = np.tanh(x)
        return out
    
    def d_activation(self, output):
        #print("TanH derivate activation function")
        out = 1 - (np.tanh(output)).pow(2)
        return out


network = [ReLuLayer(2,25),ReLuLayer(25,25),ReLuLayer(25,2)]

# initialisation parameter
learning_rate = 0.01
graph_loss = []
epochs = 50
batch_size = 1


for k in range(epochs):
    acc_loss = 0

    for i in range((int)(len(train_set)/batch_size)):
        #print(f"i : {i}")
        for layer in network:
            layer.acc_weight = 0
            layer.acc_bias = 0

        for n in range(batch_size):
            #print(f"n : {n}")
            # Foward
            output = train_set[(i*batch_size)+n]
            #print(f"input,{output}")
            for layer in network:
                output = layer.foward_pass(output)
            #print(f"output,{output}")
            #print(output)
            acc_loss = acc_loss + loss(output, train_label[(i*batch_size)+n])
            #print(f"acc_loss : {acc_loss}")
            # Backward
            dl_dy = d_loss(output, train_label[(i*batch_size)+n])
            #print(f"dl_dy : {dl_dy}")
            #print(f"output{output}, target {train_label[n]}")
            for layer in reversed(network):
                dl_dy = layer.backward_pass(dl_dy)

        # Update parameters
        output_test = test_set
        for layer in network:
            #layer.params()
            layer.update_parameters(learning_rate, batch_size)
            output_test = layer.foward_pass(output_test)

    #print(output_test)
    output_to_test = (output_test >0.5).int()
    #print(output_to_test)
    error = test_error(output_to_test, test_label)
    #print(error)
    graph_loss.append(acc_loss)
    print(f"Epoch : {k+1}, Loss : {acc_loss}, Error = {error/1000.0*100.0}%")

print("Done")   