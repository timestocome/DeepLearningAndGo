# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go

from __future__ import print_function



import goboard
import gotypes
import depthprune
import pruned_go
from utils import print_board, print_move

import time


def main():
    
    board_size = 5
    game = goboard.GameState.new_game(board_size)
    
    bots = {        
        gotypes.Player.black: depthprune.DepthPrunedAgent(3, pruned_go.capture_diff),
        gotypes.Player.white: depthprune.DepthPrunedAgent(3, pruned_go.capture_diff),
    }

    
    while not game.is_over():
    
        time.sleep(0.3)  #

        print(chr(27) + "[2J")  
        print_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        
        game = game.apply_move(bot_move)


if __name__ == '__main__':
    main()


