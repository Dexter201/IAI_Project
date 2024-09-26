from pacman_module.game import Agent, Directions
from pacman_module.util import PriorityQueueWithFunction
from pacman_module.util import manhattanDistance

def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class 'pacman.GameState'

    Returns:
        A hashable key tuple.
    """

    return (
        state.getPacmanPosition(),
        state.getFood(),
        state.getNumFood(),
        tuple(state.getCapsules())
    )


class PacmanAgent(Agent):
    """Pacman agent based on the A* algorithm.
    The manhattan distance is used as heuristic function.
    """

    def __init__(self):
        super().__init__()

        self.moves = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        if self.moves is None:
            self.moves = self.astar(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def astar(self, state):
        """Given a Pacman game state, returns a list of legal moves to
        the search layout.

        Arguments:
            state: a game state. See API or class 'pacman.GameState'.

        Returns:
            A list of legal moves.
        """

        path = []
        fringe = PriorityQueueWithFunction(self.estimatedCostOfCheapestSolution)
        fringe.push((state, path))
        closed = set()

        while True:
            if fringe.isEmpty():
                return []

            priority, (current, path) = fringe.pop()

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

    def estimatedCostOfCheapestSolution(self, node):
        """Given a node, returns the estimated cost of the cheapest
        solution through the node.
        It is computed as cost = cost of past path + estimated remaining cost.

        Arguments:
            node: a tuple containing a state and the path taken to reach it
        
        Returns:
            The estimated cost
        """

        # Computation of the current path cost
        pastCost = 0
        for i in node[1]:
            if i != Directions.STOP:
                pastCost += 1
        
        # Estimation of the remaining cost
        # It's the maximum of all manhattan distances to each points
        estimatedCost = 0
        if not node[0].isWin():
            pacman = node[0].getPacmanPosition()
            for i in range(len(node[0].getFood().data)):
                for j in range(len(node[0].getFood().data[0])):
                    if node[0].getFood().data[i][j]:
                        estimatedCost += manhattanDistance(pacman, (i, j))

            if len(node[0].getCapsules()) == 0:
                estimatedCost += 10**10

        return pastCost + estimatedCost

