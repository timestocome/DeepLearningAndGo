# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go



import numpy as np

from base import Encoder
from goboard import Point


# One plane encoder
# one color is 1, other is -1, empty spots are 0
# Board contains one plane that contains player positions
class OnePlaneEncoder(Encoder):
    
    def __init__(self, board_size):
    
        self.board_width, self.board_height = board_size
        self.num_planes = 1


    def name(self):
        return 'oneplane'


    def encode(self, game_state):
        
        # zero out a board, fetch next player
        board_matrix = np.zeros(self.shape())
        next_player = game_state.next_player
        
        # for row, column on board
        for r in range(self.board_height):
            for c in range(self.board_width):
                
                # fetch point and see what, if anything is on that point
                # 0 for null, 1 for other guy, -1 for us
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)
                
                if go_string is None:
                    continue
                
                if go_string.color == next_player:
                    board_matrix[0, r, c] = 1
                
                else:
                    board_matrix[0, r, c] = -1
        
        return board_matrix

    # convert point on board to index
    def encode_point(self, point):  
        return self.board_width * (point.row - 1) + (point.col - 1)

    # convert index to (row, column)
    def decode_point_index(self, index):
        
        row = index // self.board_width
        col = index % self.board_width
        
        return Point(row=row + 1, col=col + 1)

    # total number of points on the board
    def num_points(self):
        return self.board_width * self.board_height

    # (planes, height, width)
    def shape(self):
        return (self.num_planes, self.board_height, self.board_width)


# make a board
def create(board_size):
    return OnePlaneEncoder(board_size)
