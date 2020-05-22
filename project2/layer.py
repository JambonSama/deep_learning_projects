# Import
import numpy as np
import torch

class Linear:
    """
    Linear class : create a linear layer of the neural network
    Use random distribution with mean = 0 variance = 1 to initialize weight,
    bias initialize with 0 
    Parameters :
        nb_input -- number of neuron in input
        nb_output -- number of neuron in output
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
        Parameters : 
            x -- input value
        Retrun : 
            out -- result of activation for x
        """
        out = x
        return out

    def d_activation(self, x):
        """
        Derivate of linear activation function 
        Parameters : 
            x -- value to compute the dervate activation
        Retrun : 
            1
        """
        return 1

    def forward_pass(self,x):
        """
        Compute the forward pass, activation(weight * input + bias)
        Parameters : 
            x -- input value
        Retrun : 
            out -- result of activation(weight * x + bias)
        """
        self.input_matrix = x
        self.z = torch.matmul(x, self.weight)+self.bias
        out = self.activation(self.z)
        self.output_matrix = out
        return out
    
    def backward_pass(self, dl_dy):
        """
        Compute the backward pass, compute derivative of loss, accumulate de derivative of 
        the loss in function of the weight and the bias
        Parameters : 
            dl_dy -- derivative of the loss in function of the output
        Retrun : 
            dl_dx -- derivative of the loss in function of the input
        """
        dl_dz = self.d_activation(self.z)
        dl_dx = torch.matmul(dl_dz*dl_dy,self.weight.T)
        dl_dw = torch.matmul(self.input_matrix.view(-1,1), (dl_dz*dl_dy).view(1,-1))
        dl_db = dl_dz*dl_dy
        # Gradian accumulation
        self.acc_weight = self.acc_weight + dl_dw
        self.acc_bias = self.acc_bias + dl_db
        return dl_dx
    
    def update_parameters(self, learning_rate, batch_size):
        """
        Update the weight and the bias of the layer in function of the learning rate, batch size
        and the accumulated weight and bias
        Parameters : 
            learning_rate -- learning rate of the neural network
            batch_size -- size of the batch where the gradian was accumulate
        """
        self.weight = self.weight - learning_rate * self.acc_weight/batch_size
        self.bias = self.bias - learning_rate * self.acc_bias/batch_size

    def params(self):
        """
        Print of information of the layer
        """
        print(f"Layer : {self.layer_type}")
        print(f"Nb input : {self.nb_input}")
        print(f"Nb output : {self.nb_output}")
        print(f"Weight matrix shape : {self.weight.shape}")
        print(f"Bias matrix shape : {self.bias.shape}")
        print(f"Weight: {self.weight}")
        print(f"Bias : {self.bias}")
          
class ReLu(Linear):
    """
    ReLu class (Inheritance of linear) : create a ReLu layer of the neural network
    Use 2 time the xavier distribution to initialize weight,
    bias initialize with 0 
    Parameters :
        nb_input -- number of neuron in input
        nb_output -- number of neuron in output
    """
    def __init__(self, nb_input, nb_output):
        Linear.__init__(self, nb_input, nb_output)
        self.layer_type = "ReLu"
        self.variance = 2.0*2.0 / (nb_input + nb_output)
        self.weight = np.sqrt(self.variance)*torch.randn(nb_input, nb_output)

    def activation(self, x):
        """
        ReLu activation function output = maximum between 0 and input
        Parameters : 
            x -- input value
        Retrun : 
            out -- result of activation for x
        """
        out = (x>0).int()*x
        return out
    
    def d_activation(self, x):
        """
        Derivative of ReLu activation function
        Parameters : 
            x -- value to compute the dervate activation
        Retrun : 
            out -- result of derivate activation for x
        """
        out = (x>0).int()
        return out
    
class TanH(Linear):
    """
    TanH class (Inheritance of linear) : create a TanH layer of the neural network
    Use the xavier distribution to initialize weight,
    bias initialize with 0 
    Parameters :
        nb_input -- number of neuron in input
        nb_output -- number of neuron in output
    """
    def __init__(self, nb_input, nb_output):
        Linear.__init__(self, nb_input, nb_output)
        self.layer_type = "TanH"
        self.variance = 2.0 / (nb_input + nb_output)
        self.weight = np.sqrt(self.variance)*torch.randn(nb_input, nb_output)

    def activation(self,x):
        """
        TanH activation function 
        Parameters : 
            x -- input value
        Retrun : 
            out -- result of activation for x
        """
        out = np.tanh(x)
        return out
    
    def d_activation(self, x):
        """
        Derivate of TanH activation function 
        Parameters : 
            x -- value to compute the dervate activation
        Retrun : 
            out -- result of derivate activation for x
        """
        out = 1 - (np.tanh(x)).pow(2)
        return out