# Import
import numpy as np
import torch

class NeuralNetwork:
    """
    NeuralNetwork class : create a neural network, has to be configure with sequential
    before using it
    """
    def __init__(self):
        self.layers = []
        self.graph_loss = []
    
    def sequential(self,*args):
        """
        Construct the neural network with layer
        Parameters : 
            *args -- constructor of each layer needed for the neural network
        """
        for arg in args:
            self.layers.append(arg)

    def run(self, input_data):
        """
        Compute the output of the neural network 
        Parameters : 
            input_data -- input value
        Return : 
            output -- result of neural network
        """
        output = input_data
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output
    
    def loss(self, x, target):
        """
        Return the MSE loss between x and target
        Parameters : 
            x -- input value
            target -- target value
        Return : 
            out -- MSE loss between x and target
        """
        out = torch.mean((target-x).pow(2),0)
        return out
    
    def d_loss(self, x, target):
        """
        Return the derivative of MSE loss between x and target
        Parameters : 
            x -- input value
            target -- target value
        Return : 
            out -- derivative of MSE loss between x and target
        """
        out = 2.0 * (x - target)/len(x)
        return out
        

    def test_error(self, test_input, test_label):
        """
        Compute the number of error output of the neural network and the a label
        Parameters : 
            test_input -- input value before passing in the neural network
            test_label -- Label to compare to the output of the neural network
        Return : 
            error -- number of error of classification
        """
        output = test_input
        for layer in self.layers:
            output = layer.forward_pass(output)
        _, label = output.max(dim=1)
        error = np.sum([0 if x != y else 1 for x,y in zip(label, test_label)])
        return error

    def params(self):
        """
        Print information of the neural network
        """
        print("**** Neural network parameters ****")
        for i, layer in enumerate(self.layers):
            print(f"Layer : {i+1}")
            layer.params()

    def train_network(self, train_set, train_output, epochs, batch_size, learning_rate, print_error=False, \
                      test_set=None, test_label=None):
        """
        Train neural network
        Parameters : 
            train_set -- Set of input to train the neutal network matrix(nb, dim)
            train_output -- Set of output to compute the loss matrix(nb, dim)
            epochs -- Number of time to use train_set to train the neural network
            batch_size -- Number of input to take in acompt before updating the parameters must be a 
                          integer division from epochs
            learning_rate -- Learning rate of the neural network
            print_error -- Boolean, default False, True print the loss and % of error at each epochs
            test_set -- Default none, have to be set if print_error=True, set of input to test the neural network
            test_label -- Defaulf none, have to be set if print_error=True, set of label to test the neural 
                          network
        """
        if print_error:
            self.logs = np.zeros((epochs,3))
            
        for k in range(epochs):
            acc_loss = 0

            for i in range((int)(train_set.size(0)/batch_size)):
                
                for n in range(batch_size):
                    # Forward
                    output = self.run(train_set[(i*batch_size)+n])
                    
                    # Loss Calculus
                    acc_loss = acc_loss + self.loss(output, train_output[(i*batch_size)+n])
                    
                    # Backward
                    dl_dy = self.d_loss(output, train_output[(i*batch_size)+n])
                    for layer in reversed(self.layers):
                        dl_dy = layer.backward_pass(dl_dy)

                # Update parameters
                for layer in self.layers:
                    layer.update_parameters(learning_rate, batch_size)
                    layer.acc_weight = 0
                    layer.acc_bias = 0
            # Log loss
            self.graph_loss.append(acc_loss)

            if print_error:
                error= self.test_error(test_set, test_label)
                print(f"Epoch : {k+1}, Loss : {acc_loss:6.2f}, Error : {error/test_set.size(0)*100.0:6.2f}%")
                # Save data
                self.logs[k,0] = k+1
                self.logs[k,1] = acc_loss
                self.logs[k,2] = 100.0-error/test_set.size(0)*100.0