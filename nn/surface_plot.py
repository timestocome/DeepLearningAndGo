# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go




import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from mpl_toolkits.mplot3d import Axes3D



def plot_weights(x_len, y_len, data):
    
    
    print('Plot weights')
    print('x', x_len, np.min(data), np.max(data))
    print('y', y_len)
    print('z', data.shape)




    # Initialize figure
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')


    low = int(np.min(data)) 
    high = int(np.max(data)) 
    
    # data
    X = np.linspace(low, high, y_len)
    Y = np.linspace(low, high, x_len)
    X, Y = np.meshgrid(X, Y)
    Z = data
    
    
    print('edges', low, high)
    
    
    print('??????????????????????')
    
    print('x', X.shape)
    print('y', Y.shape)
    print('Z', Z.shape)
    
    
    '''
    # Make data.
    X = np.arange(-2, 2, 0.3)
    Y = np.arange(-2, 2, 0.3)
    X, Y = np.meshgrid(X, Y)
    R = Y * np.sin(X) - X * np.cos(Y)
    Z = np.sin(R)
    
 
    print('x', X.shape)
    print('y', Y.shape)
    print('Z', Z.shape)
    '''



    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)





    # Customize the z axis.
    ax.set_zlim(low, high)
    ax.zaxis.set_major_locator(LinearLocator(8))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))


    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


    # Show plot
    plt.show()


#plot_weights(1,1,1)