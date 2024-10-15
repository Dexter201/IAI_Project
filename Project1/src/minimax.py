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
            state: a game state. See API or class pacman.GameState.

        Returns:
            A legal move as defined in game.Directions.
        """

        self.move = minimaxSearch(state)[1]

        if self.move is not None:
            return self.move
        else:
            return Directions.STOP

def minimaxSearch(s):
    """Given a Pacman game state, returns a list [Score, Move] if pacman use the path indicated by minimax's algo.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        The best move and its score : [Score, Move].
    """

    #we know that it's Pacman which plays first
    return maxSearch(s,alpha=np.NINF, beta=np.PINF)

def maxSearch(s, alpha=0, beta=0):
    PacmanSuccessors = s.generatePacmanSuccessors()
    NextScores       = np.zeros(shape=len(PacmanSuccessors))
    NextActions      = []
    Score = np.NINF
    index = 0
    for Successor in PacmanSuccessors:
        NextState  = Successor[0]
        NextAction = Successor[1]
        if(NextState.isWin() or NextState.isLose()):                            # = terminal_test(s)
            NextScores[index]  = NextState.getScore()                           # = utility(s)
        else:
            NextScores[index]  = np.max(minSearch(NextState, alpha, beta)[0])    # = max(MINIMAX(result(s, a) = NextState))
        Score = NextScores[index]
        NextActions.extend([NextAction])
        if Score >= beta:
            indexMaxScore = np.where(NextScores==np.max(NextScores))[0][0]
            return NextScores[indexMaxScore], NextActions[indexMaxScore]
        else:
            alpha = max(Score, alpha)
        index += 1
    
    indexMaxScore = np.where(NextScores==np.max(NextScores))[0][0]
    return NextScores[indexMaxScore], NextActions[indexMaxScore]
            
def minSearch(s, alpha=0, beta=0):
    GhostSuccessors = s.generateGhostSuccessors(1)
    NextScores       = np.zeros(shape=len(GhostSuccessors))
    NextActions      = []
    index = 0
    for Successor in GhostSuccessors:
        NextState  = Successor[0]
        NextAction = Successor[1]
        if(NextState.isWin() or NextState.isLose()):                            # = terminal_test(s)
            NextScores[index] = NextState.getScore()                            # = utility(s)
        else:
            NextScores[index] = np.min(maxSearch(NextState, alpha, beta)[0])     # = min(MINIMAX(result(s, a) = NextState))
        Score = NextScores[index]
        NextActions.extend([NextAction])
        if Score <= alpha:
            indexMaxScore = np.where(NextScores==np.max(NextScores))[0][0]
            return NextScores[indexMaxScore], NextActions[indexMaxScore]
        else:
            beta = min(Score, beta)
        index += 1

    indexMinScore = np.where(NextScores==np.min(NextScores))[0][0]
    return NextScores[indexMinScore], NextActions[indexMinScore]
