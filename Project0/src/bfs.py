from pacman_module.game import Agent, Directions
from pacman_module.util import Stack


class PacmanAgent(Agent):
    "Pacman agent based on breadth first search"
    
    def __init__(self):
        super().__init__()

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A legal move as defined in `game.Directions`.
        """
