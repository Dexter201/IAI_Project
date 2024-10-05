from pacman_module.game import Agent, Directions
from pacman_module.util import Queue


def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """

    return (
        state.getPacmanPosition(),
        state.getFood(),
        state.getNumFood(),
        tuple(state.getCapsules()), 
    )


class PacmanAgent(Agent):
    """Pacman agent based on breath-first search (BFS)."""

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

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def bfs(self, state):

        path = []
        fringe = Queue()
        fringe.push((state, path))
        closed = set()

        while True:
            if fringe.isEmpty():
                return []

            current, path = fringe.pop()

            if current.isWin():
                return path

            current_key = key(current)

            if current_key in closed:
                continue
            else:
                closed.add(current_key)

            for successor, action in current.generatePacmanSuccessors():
                fringe.push((successor, path + [action]))

        return path
