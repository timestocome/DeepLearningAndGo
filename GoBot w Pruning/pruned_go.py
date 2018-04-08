# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go

from six.moves import input

import goboard
import gotypes
import depthprune
from utils import print_board, print_move, point_from_coords



BOARD_SIZE = 5


"""Calculate the difference between the number of black stones and
    white stones on the board. This will be the same as the difference
    in the number of captures, unless one player passes early.

    Returns the difference from the perspective of the next player to
    play.
    If it's black's move, we return (black stones) - (white stones).
    If it's white's move, we return (white stones) - (black stones).
"""

def capture_diff(game_state):

 
    black_stones = 0
    white_stones = 0
    
    for r in range(1, game_state.board.num_rows + 1):
        for c in range(1, game_state.board.num_cols + 1):

            p = gotypes.Point(r, c)
            color = game_state.board.get(p)

            if color == gotypes.Player.black:
                black_stones += 1

            elif color == gotypes.Player.white:
                white_stones += 1

    diff = black_stones - white_stones

    if game_state.next_player == gotypes.Player.black:
        return diff

    return -1 * diff



def main():

    game = goboard.GameState.new_game(BOARD_SIZE)
    bot = depthprune.DepthPrunedAgent(3, capture_diff)

    while not game.is_over():

        print_board(game.board)

        if game.next_player == gotypes.Player.black:
            human_move = input('-- ')
            point = point_from_coords(human_move.strip())
            move = goboard.Move.play(point)

        else:
            move = bot.select_move(game)

        print_move(game.next_player, move)
        game = game.apply_move(move)


if __name__ == '__main__':
    main()
