# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go


from __future__ import print_function
import numpy as np



# activation function to add non-linearity
def sigmoid_double(x):
    return 1.0 / (1.0 + np.exp(-x))



# vectorized version of sigmoid function
def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)


def sigmoid_prime_double(x):
    return sigmoid_double(x) * (1 - sigmoid_double(x))


def sigmoid_prime(z):
    return np.vectorize(sigmoid_prime_double)(z)





# stack layers to build forward feed network
'''
     and its
     Each layer has a forward function
    that emits output data from input data and a backward
    function that emits an output delta, i.e. a gradient,
    from an input delta.
    '''
class Layer(object):  
    
    def __init__(self):
        self.params = []

        # hook to previous layer 
        self.previous = None
        
        # hook to next layer
        self.next = None  

        # forward pass previous and next data
        self.input_data = None  
        self.output_data = None

        # backward pass previous and next data
        self.input_delta = None  
        self.output_delta = None


    # holds layer connections info
    def connect(self, layer):  
        self.previous = layer
        layer.next = self


    # abstract functions
    def forward(self):  
        raise NotImplementedError


    def get_forward_input(self):  
        if self.previous != None:
            return self.previous.output_data
        else:
            return self.input_data


    def backward(self):  
        raise NotImplementedError


    def get_backward_input(self):  
        if self.next != None:
            return self.next.output_delta
        else:
            return self.input_delta


    def clear_deltas(self):  
        pass


    def update_params(self, learning_rate):  
        pass


    def describe(self):  
        raise NotImplementedError



# sigmoid activation function for layers
class ActivationLayer(Layer):  
    
    
    def __init__(self, input_dim):
        super(ActivationLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim

    
    def forward(self):
        data = self.get_forward_input()
        self.output_data = sigmoid(data)  # <2>

    
    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        self.output_delta = delta * sigmoid_prime(data)  # <3>


    def describe(self):
        print("|-- " + self.__class__.__name__)
        print("  |-- dimensions: ({},{})"
              .format(self.input_dim, self.output_dim))
        


# fully connected layer
class DenseLayer(Layer):

    def __init__(self, input_dim, output_dim):  

        super(DenseLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = np.random.randn(output_dim, input_dim)  
        self.bias = np.random.randn(output_dim, 1)

        self.params = [self.weight, self.bias]  

        self.delta_w = np.zeros(self.weight.shape)  
        self.delta_b = np.zeros(self.bias.shape)


    # move data from previous layer forward
    def forward(self):
        data = self.get_forward_input()
        self.output_data = np.dot(self.weight, data) + self.bias  


    # move error correction backwards
    def backward(self):
        
        # input to this layer
        data = self.get_forward_input()
        
        # error
        delta = self.get_backward_input()

        # bias adjustment
        self.delta_b += delta  

        # weight adjustment
        self.delta_w += np.dot(delta, data.transpose())

        # error for previous input layer
        self.output_delta = np.dot(self.weight.transpose(), delta)  


    # adjust weights and bias
    def update_params(self, rate):  
        self.weight -= rate * self.delta_w
        self.bias -= rate * self.delta_b

    # zero out error adjustments before next pass
    def clear_deltas(self):  
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)


    def describe(self):  # <3>
        print("|--- " + self.__class__.__name__)
        print("  |-- dimensions: ({},{})"
              .format(self.input_dim, self.output_dim))

