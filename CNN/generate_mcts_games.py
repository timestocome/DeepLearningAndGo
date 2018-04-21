# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go


# use to create several game simulations


import numpy as np

from base import get_encoder_by_name
import goboard
import mcts
#from utils import print_board, print_move


# generate mcts games
def generate_game(board_size, rounds, max_moves, temperature):
    
    # init
    boards = []  # encoded board states
    moves = []   # encoded moves
    
    # only use player position as inputs to nn
    encoder = get_encoder_by_name('oneplane', board_size)
    game = goboard.GameState.new_game(board_size)  
    bot = mcts.MCTSAgent(rounds, temperature)  

    # run
    num_moves = 0
    while not game.is_over():
        
        #print_board(game.board)
        
        move = bot.select_move(game)  
        if move.is_play:
            
            # play move, update data
            boards.append(encoder.encode(game))
            move_one_hot = np.zeros(encoder.num_points())
            move_one_hot[encoder.encode_point(move.point)] = 1
            moves.append(move_one_hot)

        #print_move(game.next_player, move)
        game = game.apply_move(move)  
        
        num_moves += 1
        
        # this game is going no where
        if num_moves > max_moves:  
            break

    return np.array(boards), np.array(moves)

###############################################################################
# run code
###############################################################################

# run games
def main():
    
    # defaults
    board_size = 9          # n x n go go board
    rounds = 100              # mcts n_moves
    temperature = 0.8       # c * sqrt(log(N/n)) c is temperature, higher explores more
    max_moves = 50          # max moves per game
    n_games = 30            # number of games
    board_out = 'board_out'
    move_out = 'labels.npy'

    boards = []
    moves = []


    for i in range(n_games):

        print('Generating game %d/%d...' % (i + 1, n_games))
        bs, ms = generate_game(board_size, rounds, max_moves, temperature)  
        boards.append(bs)
        moves.append(ms)
        
        print('boards', len(bs))
        print('moves', len(ms))     

    # save data
    all_boards = np.concatenate(bs)  
    all_moves = np.concatenate(ms)
    np.save(board_out, all_boards)  
    np.save(move_out, all_moves)

###############################################################################
if __name__ == '__main__':
    main()
