from pacman_module.game import Agent
from collections import deque
import numpy as np


def key(state):
    """Returns a key that uniquely identifies a pacman game state.

    Arguments:
        state: a game state, See API or class 'pacman.GameState'

    Returns:
        A hashable key tuple.
    """

    ghostPositions = []
    agentIndex = 1
    pos = (0, 0)
    while (pos is not None):
        try:
            pos = state.getGhostPosition(agentIndex)
        except IndexError:
            pos = None

        if pos is not None:
            ghostPositions.append(pos)
        agentIndex += 1

    return (state.getPacmanPosition(),
            tuple(ghostPositions), state.getFood())


class PacmanAgent(Agent):

    def __init__(self):
        super().__init__()

        # storing crutial information
        # partly only calculated once for increased performance
        self.move = None
        self.numberOfFood = None
        self.MatrixPath = None
        self.VerticesSet = None
        self.VerticesDic = None
        self.numberOfSameStates = 0
        self.state_history = deque(maxlen=20)

        # changable parameters
        self.utilityParameters = [1, 5, 1, 10]
        self.utilityParameter_MeanFoodDist = 10
        self.maxRecursionDepth = 10
        self.maxNumberOfSameStates = 3

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        if self.numberOfFood is None:
            self.numberOfFood = state.getNumFood()

        if self.MatrixPath is None:
            self.VerticesSet = set()
            self.VerticesDic = dict()
            self.InitMatrixPath(state)

        self.move = self.hminimaxSearch(state)[1]
        return self.move

    def hminimaxSearch(self, state):
        """Given a Pacman game state, returns a list [Score, Move]
        if pacman use the path indicated by minimax's algo.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The best move and its score : [Score, Move].
        """

        # we know that it's Pacman which plays first

        self.closed = set()
        self.values = dict()
        depth = 0
        return self.hmaxSearch(state, depth)

    def hmaxSearch(self, state, depth):
        """Given a Pacman game state, return s the favorable score for
        a maximisation in the H-minimax algorithm.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.
            depth: the current depth of the search.

        Returns:
            The favorable score.
        """
        depth += 1
        current_key = key(state)
        if current_key in self.closed:
            return self.values[current_key]

        PacmanSuccessors = state.generatePacmanSuccessors()
        NextScores = np.zeros(shape=len(PacmanSuccessors))
        NextActions = []
        index = 0
        for Successor in PacmanSuccessors:
            NextState = Successor[0]
            NextAction = Successor[1]
            if self.cutoffTest(NextState, depth):   # = terminal_test(s)
                NextScores[index] = self.utility(
                    NextState, state)   # = utility(s)
                NextActions.append(NextAction)
            else:
                NextScores[index] = np.max(
                    # = max(MINIMAX(result(s, a) = NextState))
                    self.hminSearch(NextState, depth)[0])
                NextActions.append(NextAction)
            index += 1

        indexMaxScore = np.where(
            NextScores == np.max(NextScores))[0][0]

        self.closed.add(current_key)
        self.values[current_key] = [NextScores[indexMaxScore],
                                    NextActions[indexMaxScore]]
        depth -= 1
        return NextScores[indexMaxScore], NextActions[indexMaxScore]

    def hminSearch(self, state, depth):
        """Given a Pacman game state, return s the favorable score for
        a minimisation in the H-minimax algorithm.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.
            depth: the current depth of the search.

        Returns:
            The favorable score.
        """
        depth += 1
        current_key = key(state)
        if current_key in self.closed:
            return self.values[current_key]

        GhostSuccessors = state.generateGhostSuccessors(1)
        NextScores = np.zeros(shape=len(GhostSuccessors))
        NextActions = []
        index = 0
        for Successor in GhostSuccessors:
            NextState = Successor[0]
            NextAction = Successor[1]
            if self.cutoffTest(NextState, depth):    # = terminal_test(s)
                NextScores[index] = self.utility(NextState)    # = utility(s)
                NextActions.append(NextAction)
            else:
                NextScores[index] = np.min(
                    # = min(MINIMAX(result(s, a) = NextState))
                    self.hmaxSearch(NextState, depth)[0])
                NextActions.append(NextAction)
            index += 1

        indexMinScore = np.where(NextScores == np.min(NextScores))[0][0]

        self.closed.add(current_key)
        self.values[current_key] = [NextScores[indexMinScore],
                                    NextActions[indexMinScore]]
        depth -= 1
        return (NextScores[indexMinScore],
                NextActions[indexMinScore])

    def cutoffTest(self, state, depth):
        """Given a state, returns whether
        the hminimax algorithm should stop there.

        Arguements:
            state: a state

        Returns:
            A boolean, True is the evaluation has to happen there.
        """

        stop = False
        if ((depth > self.maxRecursionDepth)
                or state.isWin() or state.isLose()):
            stop = True

        return stop

    def utility(self, state):
        """
        Calculate the utility value of a
        game state based on Pacman's position,
        food distances, ghost distances, and recency penalties.

    Args:
        state (State): Current game state,
        providing information like Pacman's position,
        food layout, and ghost locations.
        previousState (State): Previous game state,
        used for comparing certain conditions.

    Returns:
        float: The calculated utility score for the current
        game state based on distance to food,
        proximity to ghosts, and recency penalties.
    """

        DistGhostPacmanList = []
        pacmanPosition = state.getPacmanPosition()

        # LET'S GET THE MIN DIST FROM PACMAN TO EACH FOOD
        DistPacmanFoodList = []

        foodMatrix = state.getFood().data
        sizeFoodMatrixX = len(foodMatrix)
        sizeFoodMatrixY = len(foodMatrix[0])
        for PosX in range(sizeFoodMatrixX):
            for PosY in range(sizeFoodMatrixY):
                foodPosition = (PosX, PosY)
                if foodMatrix[PosX][PosY]:
                    keyVerticesDic = (pacmanPosition, foodPosition)
                    indexI_indexJ = self.VerticesDic[keyVerticesDic]

                    indexI = indexI_indexJ[0]
                    indexJ = indexI_indexJ[1]

                    minDistPacmanFood = self.MatrixPath[indexI][indexJ]
                    DistPacmanFoodList.append(minDistPacmanFood)

        # LET'S GET ALL THE GHOST POSITION
        GhostPositionList = []
        agentIndex = 1
        pos = (0, 0)
        while pos is not None:
            try:
                pos = state.getGhostPosition(agentIndex)
            except IndexError:
                pos = None
            if pos is not None:
                agentLocation = state.getGhostPosition(agentIndex)
                GhostPositionList.append(agentLocation)
            agentIndex += 1

        # LET'S GET THE MIN DIST FROM PACMAN TO EACH GHOST
        for agentIndex, GhostPos in enumerate(GhostPositionList):

            keyVerticesDic = (GhostPos, pacmanPosition)
            indexI_indexJ = self.VerticesDic[keyVerticesDic]

            indexI = indexI_indexJ[0]
            indexJ = indexI_indexJ[1]

            minDistGhostPacman = self.MatrixPath[indexI][indexJ]

            DistGhostPacmanList.append(minDistGhostPacman)

        # LET'S COMPUTE UTILITY
        FoodTerm = 0
        GhostTerm = 0
        if state.getNumFood() != 0:
            FoodTerm = (((np.min(DistPacmanFoodList)) /
                        (self.numberOfFood)) +
                        self.utilityParameter_MeanFoodDist *
                        (np.sum(DistPacmanFoodList)) /
                        (self.numberOfFood)) * state.getNumFood()

            if (np.min(DistGhostPacmanList) != 0.0):
                GhostTerm = ((1/np.min(DistGhostPacmanList)))
            else:
                GhostTerm = np.inf

        recencyPenalty = self.getSameStatePenalty(state)
        utilityScores = ([
                            state.getScore(),
                            - FoodTerm,
                            - GhostTerm,
                            - recencyPenalty])
        utility = np.dot(
                        utilityScores,
                        self.utilityParameters)

        return utility

    # ----------------------- TOOLS FUNCTIONS ------------------------- #
    """
    The idea of this file's implementation is to
    compute a huge matrix within all
    the distance from one clear position of
    the game's grid to any other position

    For this we use a self variable dubbed MatrixPath
        MatrixPath[vertex1][vertex2]
        -> returns the distance from
        vertex1 to vertex2 in the game's grid
        -> returns np.inf if one of the vertex is a wall

    The Matrix is initialized according to
    Floyd-Warshall algorithm in the function
    'InitMatrixPath'

    The algortihm is then performed in
    the 'Floyd_Warshall' function and this handle
    self.MatrixPath such that at the end
    the matrix contains the right values

    This is quite usefull once it has been done because
    it enables to reach a good complexity
    during the game computations for utility
    """

    def InitMatrixPath(self, state):
        """
        Initialize the distance matrix (`MatrixPath`)
        that records the shortest paths between
        vertices in the game's grid using
        the Floyd-Warshall algorithm.

        Args:
            state (State): Initial game state to determine
            wall positions and grid dimensions.

        Returns:
            None: Sets up `MatrixPath` and `VerticesDic`
            for efficient access during game computations.
        """
        sizeMazeX = len(state.getFood().data)
        sizeMazeY = len(state.getFood().data[0])

        for i in range(sizeMazeX):
            for j in range(sizeMazeY):
                self.VerticesSet.add((i, j))

        nbVertices = sizeMazeX * sizeMazeY
        self.MatrixPath = np.full(
            (nbVertices, nbVertices), fill_value=np.inf)
        for indexI, vertexFrom in enumerate(self.VerticesSet):
            for indexJ, vertexTo in enumerate(self.VerticesSet):
                if (
                    self.isWall(state, vertexFrom)
                        or self.isWall(state, vertexTo)):
                    self.MatrixPath[indexI][indexJ] = np.inf

                elif vertexFrom == vertexTo:
                    self.MatrixPath[indexI][indexJ] = 0

                elif self.isNeighbour(vertexFrom, vertexTo):
                    self.MatrixPath[indexI][indexJ] = 1

                keyVerticesDic = (vertexFrom, vertexTo)
                self.VerticesDic[keyVerticesDic] = (indexI, indexJ)

        self.Floyd_Warshall()
        return

    def isWall(self, state, vertex):
        """Check if vertex is wall or not

        Args:
            state (state): a state describing what the game looks like
            vertex (pair): coordinate of a position in the game's grid

        Returns:
            bool: True if vertex is Wall and False otherwise
        """
        wallMatrix = state.getWalls().data
        if wallMatrix[vertex[0]][vertex[1]]:
            return True
        else:
            return False

    def isNeighbour(self, vertex1, vertex2):
        """check if these vertices are neighbours

        Args:
            vertex1 (pair): coordinate
            of a position in the game's grid
            vertex2 (pair): coordinate
            of a position in the game's grid

        Returns:
            bool: True if vertex1 is a neighbour
            of vertex2 and False otherwise
        """

        if (((vertex1[0] == vertex2[0] + 1
            or vertex1[0] == vertex2[0] - 1))
                and vertex1[1] == vertex2[1]):
            return True

        if (((vertex1[1] == vertex2[1] + 1
            or vertex1[1] == vertex2[1] - 1))
                and vertex1[0] == vertex2[0]):
            return True
        else:
            return False

    def Floyd_Warshall(self):
        """
        Apply the Floyd-Warshall algorithm to calculate the shortest
        paths between all pairs of vertices in the game's grid.

        Returns:
            None: Updates `MatrixPath` with shortest distances
            between all reachable pairs of vertices in the grid.
        """
        for vertexIntermediate in self.VerticesSet:
            for vertexFrom in self.VerticesSet:
                for vertexTo in self.VerticesSet:
                    keyVerticesDicIJ = (vertexFrom, vertexTo)
                    keyVerticesDicIK = (vertexFrom, vertexIntermediate)
                    indexI_indexJ = self.VerticesDic[keyVerticesDicIJ]
                    indexI_indexK = self.VerticesDic[keyVerticesDicIK]

                    if indexI_indexJ[0] != indexI_indexK[0]:
                        raise (ValueError(
                            "Error in Floyd_Warshall"))

                    indexI = indexI_indexK[0]
                    indexJ = indexI_indexJ[1]
                    indexK = indexI_indexK[1]

                    if (self.MatrixPath[indexI][indexJ] is not None and
                            self.MatrixPath[indexI][indexK] is not None and
                            self.MatrixPath[indexK][indexJ] is not None):
                        oldLengthPath = self.MatrixPath[indexI][indexJ]
                        newLenghtPath = (
                            self.MatrixPath[indexI][indexK] +
                            self.MatrixPath[indexK][indexJ])

                        if oldLengthPath > newLenghtPath:
                            self.MatrixPath[indexI][indexJ] = newLenghtPath
                    else:
                        continue
        return

    def getSameStatePenalty(self, state):
        """
        Determine a penalty for repeated states to prevent
        loops or redundant states in the game.

        Args:
            state (State): Current game state
            to check for recurrence.

        Returns:
            int: A penalty value if the current state has been
            frequently encountered; otherwise, zero.
        """

        state_key = key(state)
        state_hash = hash((hash(state_key[0]), hash(state_key[1])))
        recencyPenalty = 0

        if state_hash in self.state_history:
            self.numberOfSameStates += 1

        if (self.numberOfSameStates > self.maxNumberOfSameStates
            and state_hash
                in self.state_history):
            recencyPenalty = self.numberOfSameStates
        elif (self.numberOfSameStates > self.maxNumberOfSameStates):
            self.numberOfSameStates = 0

        self.state_history.append(state_hash)
        return recencyPenalty
