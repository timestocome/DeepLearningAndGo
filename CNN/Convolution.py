# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go



import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


np.random.seed(42)



# load saved games from book author's github
X = np.load('features-200.npy')
Y = np.load('labels-200.npy')


# sizes
n_samples = X.shape[0]
board_size = 9
input_shape = (board_size, board_size, 1)

X = X.reshape(n_samples, board_size, board_size, 1)

# split into train/test
n_train = 10000

X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]



# create model
#  input
model = Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 activation='relu',
                 input_shape=input_shape
                 ))
model.add(Dropout(rate=0.6))

# hidden
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.6))
model.add(Flatten())

# output
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.6))
model.add(Dense(board_size * board_size, activation='softmax'))

model.summary()



# build model
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


# train model
model.fit(X_train, Y_train,
          batch_size=64,
          epochs=5,
          verbose=1,
          validation_data=(X_test, Y_test))


# test model
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])











