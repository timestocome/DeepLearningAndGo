# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go


# attempt to solve MNIST by averaging number images

import numpy as np
from load_mnist import load_data
from layers import sigmoid_double
from matplotlib import pyplot as plt



# compute average over all samples in training set
def average_digit(data, digit):  
        
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    
    return np.average(filtered_array, axis=0)


train, test = load_data()



# train, test on digit 8
avg_eight = average_digit(train, 8)  


# image showing 8 average
img = (np.reshape(avg_eight, (28, 28)))
plt.imshow(img)
plt.show()





# test a few random samples
x_3 = train[2][0]    
x_18 = train[17][0]  

# check distance between average 8 and random samples
W = np.transpose(avg_eight)
np.dot(W, x_3)
np.dot(W, x_18)  




# make predictions
def predict(x, W, b):  
    return sigmoid_double(np.dot(W, x) + b)


b = -45  # set bias to -45 based on test samples

print('predictions....')
print(predict(x_3, W, b))   
print(predict(x_18, W, b))  
print('        ')


# test accuracy
def evaluate(data, digit, threshold, W, b):  
    
    total_samples = 1.0 * len(data)
    correct_predictions = 0
    
    # for each image
    for x in data:
        # test if over / under threshold to match '8'
        if predict(x[0], W, b) > threshold and np.argmax(x[1]) == digit:  # <2>
            correct_predictions += 1
        
        if predict(x[0], W, b) <= threshold and np.argmax(x[1]) != digit:  # <3>
            correct_predictions += 1
            
            
    accuracy = correct_predictions / total_samples
    print('Accuracy  ', accuracy)
    print('      ')
    
    return correct_predictions / total_samples



# check accuracy on training and testing samples
print('Test training data...')
evaluate(data=train, digit=8, threshold=0.5, W=W, b=b)  

print('Test test data...')
evaluate(data=test, digit=8, threshold=0.5, W=W, b=b)

# check accuracy on '8's only
print('Only test digit 8 samples.....')
eight_test = [x for x in test if np.argmax(x[1]) == 8]
evaluate(data=eight_test, digit=8, threshold=0.5, W=W, b=b) 