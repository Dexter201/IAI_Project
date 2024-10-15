from pacman_module.game import Agent, Directions
import numpy as np

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
        if self.move is None:
            self.move = minimaxSearch(state)[1]
        
        if self.move is not None:
            return self.move
        else:
            return Directions.STOP

def minimaxSearch(state):
    """Given a Pacman game state, returns a list [Score, Move] if pacman use the path indicated by minimax's algo.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        The best move and its score : [Score, Move].
    """

    #we know that it's Pacman which plays first
    return maxSearch(state)

def maxSearch(state):
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
            NextScores[index]  = np.max(minSearch(NextState)[0])    # = max(MINIMAX(result(s, a) = NextState))
            NextActions.append(NextAction)
        index += 1
    
    indexMaxScore = np.where(NextScores==np.max(NextScores))[0][0]
    return NextScores[indexMaxScore], NextActions[indexMaxScore]
            
def minSearch(state):
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
            NextScores[index] = np.min(maxSearch(NextState)[0])     # = min(MINIMAX(result(s, a) = NextState))
            NextActions.append(NextAction)
        index += 1

    indexMinScore = np.where(NextScores==np.min(NextScores))[0][0]
    return NextScores[indexMinScore], NextActions[indexMinScore]
