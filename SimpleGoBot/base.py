# http://github.com/timestocome

# adapted from:
#  https://github.com/maxpumperla/betago
#  https://www.manning.com/books/deep-learning-and-the-game-of-go

__all__ = [
    'Agent',
]



class Agent():

    """Interface for a go-playing bot."""
    def select_move(self, game_state):
        raise NotImplementedError()


    def diagnostics(self):
        return {}
