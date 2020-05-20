#!/usr/bin/env python
# ne pas enlever la ligne 1, ne pas mettre de commentaire au dessus

# Import
import numpy as np
import torch
torch.set_grad_enabled(False)

# Class definition
class LinearLayer:
    """
    Class
    decription
    parameters


    """
    def __init__(self, nb_input, nb_output):
        self.layer_type = "Linear"
        self.variance = 1
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.weight = np.sqrt(self.variance)*torch.randn(nb_input, nb_output)
        self.bias = torch.zeros(nb_output)

    def activation(self,x):
        """
        Linear activation function output = input
        Parameters : x input value
        Retrun : out result of activation for x
        """
        out = x
        return out

    def d_activation(self, output):
        return 1

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
        # Accumulation parameter
        self.acc_weight = self.acc_weight + dl_dw
        self.acc_bias = self.acc_bias + dl_db
        return dl_dx
    
    def update_parameters(self, learning_rate, batch_size):
        self.weight = self.weight - learning_rate * self.acc_weight/batch_size
        self.bias = self.bias - learning_rate * self.acc_bias/batch_size
        
    def params(self):
        print(f"Layer : {self.layer_type}")
        print(f"Nb input : {self.nb_input}")
        print(f"Nb output : {self.nb_output}")
        print(f"Weight matrix shape : {self.weight.shape}")
        print(f"Bias matrix shape : {self.bias.shape}")
        print(f"Weight: {self.weight}")
        print(f"Bias : {self.bias}")
          
class ReLuLayer(LinearLayer):
    def __init__(self, nb_input, nb_output):
        LinearLayer.__init__(self, nb_input, nb_output)
        self.layer_type = "ReLu"
        self.variance = 1.0 * 2.0 / (nb_input + nb_output)
        self.weight = np.sqrt(self.variance)*torch.randn(nb_input, nb_output)

    def activation(self,x):
        """
        ReLu activation function output = maximum between 0 and input
        Parameters : x input value
        Retrun : out result of activation for x
        """
        out = (x>0).int()*x
        return out
    
    def d_activation(self, output):
        #print("ReLu derivate activation function")
        out = (output>0).int()
        return out
    
class TanHLayer(LinearLayer):
    def __init__(self, nb_input, nb_output):
        LinearLayer.__init__(self, nb_input, nb_output)
        self.layer_type = "TanH"
        self.variance = 2.0 / (nb_input + nb_output)
        self.weight = np.sqrt(self.variance)*torch.randn(nb_input, nb_output)

    def activation(self,x):
        """
        TanH activation function output = 
        Parameters : 
            x -- input value
        Retrun : 
            out -- result of activation for x
        """
        out = np.tanh(x)
        return out
    
    def d_activation(self, output):
        #print("TanH derivate activation function")
        out = 1 - (np.tanh(output)).pow(2)
        return out


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.graph_loss = []

    def sequential(self,*args):
        for arg in args:
            self.layers.append(arg)

    def run(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.foward_pass
        return output
    
    def loss(self, x, target):
        out = torch.mean((target-x).pow(2),0)
        return out
    
    def d_loss(self, output, target):
        out = 2.0 * (output - target)/len(output)
        return out
        

    def test_error(self, test_input, test_label):
        output = test_input
        for layer in self.layers:
            output = layer.foward_pass(output)
        _, test2 = output.max(dim=1)
        error = np.sum([0 if x != y else 1 for x,y in zip(test2, test_label)])
        return error

    def params(self):
        print("****Neural network parameters****")
        for i, layer in enumerate(self.layers):
            print(f"Layer : {i+1}")
            layer.params()

    def train_network(self, train_set, train_label, epochs, batch_size, learning_rate, print_error=False, test_set=None, test_label=None):
        for k in range(epochs):
            acc_loss = 0

            for i in range((int)(train_set.size(0)/batch_size)):
                
                for layer in self.layers:
                    layer.acc_weight = 0
                    layer.acc_bias = 0

                for n in range(batch_size):
                    # Foward
                    output = train_set[(i*batch_size)+n]
                    for layer in self.layers:
                        output = layer.foward_pass(output)
                    
                    # Loss Calculus
                    acc_loss = acc_loss + self.loss(output, train_label[(i*batch_size)+n])
                    
                    # Backward
                    dl_dy = self.d_loss(output, train_label[(i*batch_size)+n])
                    for layer in reversed(self.layers):
                        dl_dy = layer.backward_pass(dl_dy)

                # Update parameters
                for layer in self.layers:
                    layer.update_parameters(learning_rate, batch_size)
            # Log loss
            self.graph_loss.append(acc_loss)

            if print_error:
                error= self.test_error(test_set, test_label)
                print(f"Epoch : {k+1}, Loss : {acc_loss:6.2f}, Error : {error/test_set.size(0)*100.0:6.2f}%")
            
            
def main():
    # Parameters for train and test set generation
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
    net = NeuralNetwork()
    net.sequential(ReLuLayer(2,25), ReLuLayer(25,25), ReLuLayer(25,25), ReLuLayer(25,2))
    net.train_network(train_set, train_label, epochs=50, batch_size=1, learning_rate=0.02, print_error=True, test_set=test_set, test_label=test_label)
    net.params()

if __name__ == "__main__":
    main()
    print("Done")   