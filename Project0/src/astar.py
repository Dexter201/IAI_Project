from pacman_module.game import Agent, Directions
from pacman_module.util import PriorityQueueWithFunction
from pacman_module.util import manhattanDistance
import numpy as np


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
        tuple(state.getCapsules()),
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
        fringe = PriorityQueueWithFunction(
            self.estimatedCostOfCheapestSolution
        )
        fringe.push((None, state, path, 0))
        closed = set()

        while True:
            if fringe.isEmpty():
                return []

            priority, (past, current, path, cost) = fringe.pop()

            if current.isWin():
                return path

            current_key = key(current)

            if current_key in closed:
                continue
            else:
                closed.add(current_key)

            for successor, action in current.generatePacmanSuccessors():
                fringe.push(
                    (
                        current,
                        successor,
                        path + [action],
                        self.pastCost(current, successor, cost, action),
                    )
                )

        return path

    def estimatedCostOfCheapestSolution(self, node):
        """Given a node, returns the estimated cost of the cheapest
        solution through the node.
        It is computed as cost = cost of past path + estimated remaining cost.

        Arguments:
            node: a tuple containing:
                - the past state
                - the current state
                - the path taken to get to the current state
                - the total cost to get to the current state

        Returns:
            The estimated cost to win the game through this node.
        """

        # Computation of the current path cost
        pastCost = node[3]

        # Estimation of the remaining cost
        # It's the maximum of all manhattan distances to each points
        estimatedCost = self.heuristic(node[1])

        return pastCost + estimatedCost

    def pastCost(self, past, current, cost, action):
        """Given the past state, the cost it took to get to it, the current
        state, and the action made to go from the past state to the current
        one, computes the cost taken to get to the current state.

        Arguments:
            past: the state before the one that the computation is for.
            current: the state for which the cost is being computed.
            cost: the cost it took to get to the past state.
            action: the last action taken.

        Returns:
            The total cost to get from the start to the current state.
        """
        newCost = 0
        if action != Directions.STOP:
            newCost += 1
        if len(past.getCapsules()) != len(current.getCapsules()):
            newCost += 5
        if past.getNumFood() != current.getNumFood():
            newCost -= 10

        return cost + newCost

    def heuristic(self, state):
        """Given a state, returns the estimated cost to win through this state.
        Computes the mean manhattan distance to the food.

        Arguments:
            state: a state of the environment

        Returns:
            The estimated cost
        """
        estimatedCost = []

        if not state.isWin():
            pacman = state.getPacmanPosition()

            for i in range(len(state.getFood().data)):
                for j in range(len(state.getFood().data[0])):
                    if state.getFood().data[i][j]:
                        estimatedCost.append(manhattanDistance(pacman, (i, j)))

            return np.mean(estimatedCost)

        else:
            return 0
