from pacman_module.game import Agent
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
    while(pos is not None):
        try:
            pos = state.getGhostPosition(agentIndex)
        except IndexError:
            pos = None

        if pos is not None:
            ghostPositions.append(pos)
        agentIndex += 1

    return (state.getPacmanPosition(),tuple(ghostPositions), state.getFood())

class PacmanAgent(Agent):
    """Empty Pacman agent."""

    def __init__(self):
        super().__init__()

        self.move = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        self.move = minimaxSearch(state)[1]

        return self.move

def minimaxSearch(state):
    """Given a Pacman game state, returns a list [Score, Move] if pacman use the path indicated by minimax's algo.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        The best move and its score : [Score, Move].
    """

    #we know that it's Pacman which plays first

    closed = set()
    values = dict()

    return maxSearch(state, closed, values)

def maxSearch(state, closed, values):
    current_key = key(state)
    if current_key in closed:
        return values[current_key]

    PacmanSuccessors = state.generatePacmanSuccessors()
    NextScores       = np.zeros(shape=len(PacmanSuccessors))
    NextActions      = []
    index = 0
    for Successor in PacmanSuccessors:
        NextState  = Successor[0]
        NextAction = Successor[1]
        if(NextState.isWin() or NextState.isLose()):                # = terminal_test(s)
            NextScores[index]  = NextState.getScore()               # = utility(s)
            NextActions.append(NextAction)
        else:
            NextScores[index]  = np.max(minSearch(NextState, closed, values)[0])    # = max(MINIMAX(result(s, a) = NextState))
            NextActions.append(NextAction)
        index += 1
    
    indexMaxScore = np.where(NextScores==np.max(NextScores))[0][0]

    closed.add(current_key)
    values[current_key] = [NextScores[indexMaxScore], NextActions[indexMaxScore]]

    return NextScores[indexMaxScore], NextActions[indexMaxScore]
            
def minSearch(state, closed, values):
    current_key = key(state)
    if current_key in closed:
        return values[current_key]

    GhostSuccessors = state.generateGhostSuccessors(1)
    NextScores       = np.zeros(shape=len(GhostSuccessors))
    NextActions      = []
    index = 0
    for Successor in GhostSuccessors:
        NextState  = Successor[0]
        NextAction = Successor[1]
        if(NextState.isWin() or NextState.isLose()):                # = terminal_test(s)
            NextScores[index] = NextState.getScore()                # = utility(s)
            NextActions.append(NextAction)
        else:
            NextScores[index] = np.min(maxSearch(NextState, closed, values)[0])     # = min(MINIMAX(result(s, a) = NextState))
            NextActions.append(NextAction)
        index += 1

    indexMinScore = np.where(NextScores==np.min(NextScores))[0][0]

    closed.add(current_key)
    values[current_key] = [NextScores[indexMinScore], NextActions[indexMinScore]]

    return NextScores[indexMinScore], NextActions[indexMinScore]
