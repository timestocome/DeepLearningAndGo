# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go




from __future__ import print_function



import goboard
import gotypes
import mcts
from utils import print_board, print_move

import time


def main():
    
    board_size = 5
    game = goboard.GameState.new_game(board_size)

    # temperature == c in w + c*sqrt(log N / n)
    # higher temperatures are more volatile, 
    # lower temperaures create a more focused search
    temperature = 1.5  

    
    bots = {        
        gotypes.Player.black: mcts.MCTSAgent(3, temperature),
        gotypes.Player.white: mcts.MCTSAgent(3, temperature),
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


