# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go




import load_mnist
import network
from layers import DenseLayer, ActivationLayer



###############################################################################
# run network
###############################################################################



training_data, test_data = load_mnist.load_data()  


net = network.SequentialNetwork()  


# load input into network
net.add(DenseLayer(784, 392))  
net.add(ActivationLayer(392))


# hidden layer
net.add(DenseLayer(392, 196))
net.add(ActivationLayer(196))


# output layer
net.add(DenseLayer(196, 10))
net.add(ActivationLayer(10))  


# run code
net.train(training_data, epochs=1, mini_batch_size=10,
          learning_rate=3.0, test_data=test_data)  









'''

# print a 3d plot of weights between layers
# code plots input layer (784, 392)

from surface_plot import plot_weights

print('layer weights')
print(net.layers)
print('n layers', len(net.layers))

  
print('-------------------------------------------------', 0)
p = net.layers[0].params

x = p[0].shape[0]
y = p[0].shape[1]
z = p[0]
plot_weights(x, y, z)



'''