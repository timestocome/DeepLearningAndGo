# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go



import six.moves.cPickle as pickle
import gzip
import numpy as np



# convert labels (0..9) to one hot vectors
def encode_label(j):  

    e = np.zeros((10, 1))
    e[j] = 1.0

    return e


def shape_data(data, encode=True):
    
    # flatten 28x28 image to 784x1 vector
    features = [np.reshape(x, (784, 1)) for x in data[0]]  

    # convert labels to one hot vectors
    labels = [encode_label(y) for y in data[1]]  

    
    # return tuples (image, label)
    return list(zip(features, labels))  



def load_data():

    # unzip data
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        
    # keep only train, test data for this example
    return (shape_data(train_data), shape_data(test_data))

