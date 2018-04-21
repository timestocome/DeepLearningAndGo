# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go


# Abstract class
# sets up game playing bots
# Encodes Go games into a form useable by a neural network


import importlib

__all__ = [
    'Agent',
    'Encoder',
    'get_encoder_by_name',
]



# go playing bot
class Agent():

    def select_move(self, game_state):
        raise NotImplementedError()


    def diagnostics(self):
        return {}




# Base encoder
class Encoder():
    
    # load an encoder by name
    def name(self):
        raise NotImplementedError()

    # convert Go board to numberic data
    def encode(self, game_state):  
        raise NotImplementedError()

    # convert point on Go board to int (x, y)
    def encode_point(self, point):
        raise NotImplementedError()

    # convert point back to position on Go board
    def decode_point_index(self, index):  
        raise NotImplementedError()

    # height * width = number of points on board
    def num_points(self):  
        raise NotImplementedError()

    # encoded board shape
    def shape(self):  
        raise NotImplementedError()



# create an encoder using name string
def get_encoder_by_name(name, board_size):  
    
    # if board size only one int assume it's square
    if isinstance(board_size, int):
        board_size = (board_size, board_size)  
        
    # need a create function for each encoder    
    module = importlib.import_module(name)
    constructor = getattr(module, 'create')
    
    return constructor(board_size)
