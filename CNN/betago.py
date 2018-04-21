# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go


import numpy as np

from base import Encoder
from goboard import Move, Point



# 7 plane encoder
# board_size === (width, height)
class BetaGoEncoder(Encoder):

   
    def __init__(self, board_size):
        
        self.board_width, self.board_height = board_size
        # 0 - 2. our stone with 1, 2, 3+ liberties
        # 3 - 5. opponent stone with 1, 2, 3+ liberties
        # 6. move would be illegal due to ko
        self.num_planes = 7


    def name(self):
        return 'betago'


    def encode(self, game_state):

        board_tensor = np.zeros(self.shape())
        base_plane = {
            game_state.next_player: 0,
            game_state.next_player.other: 3,
        }

        for r in range(self.board_height):
            for c in range(self.board_width):

                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None:
                    if game_state.does_move_violate_ko(game_state.next_player,
                                                       Move.play(p)):
                        board_tensor[6][r][c] = 1
                else:
                    liberty_plane = min(3, go_string.num_liberties) - 1
                    liberty_plane += base_plane[go_string.color]
                    board_tensor[liberty_plane][r][c] = 1

        return board_tensor


    # convert board state to ints
    def encode_point(self, point):
        return self.board_width * (point.row - 1) + (point.col - 1)


    # convert board index to (r, c)
    def decode_point_index(self, index):

        row = index // self.board_width
        col = index % self.board_width

        return Point(row=row + 1, col=col + 1)

    # total number of points on board
    def num_points(self):
        return self.board_width * self.board_height

    # n_planes, height, width
    def shape(self):
        return (self.num_planes, self.board_height, self.board_width)


def create(board_size):
    return BetaGoEncoder(board_size)
