from pacman_module.game import Agent, Directions
from pacman_module.util import Stack
from dfs import key
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """

class PacmanAgent(Agent):
    "Pacman agent based on breadth first search"
    
    def __init__(self):
        super().__init__() 
        
        self.moves = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A legal move as defined in `game.Directions`.
        """
        
        if self.moves is None:
            self.moves = self.bfs(state)
            
        if self.moves is not None
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def bfs(self, state):
        """Given a Pacman game state, returns a list of legal moves to solve
        the search layout, by the depth first search strategy.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A list of legal moves.
        """
        
        