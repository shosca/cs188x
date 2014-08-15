# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
import sys
from util import manhattanDistance
from game import Directions
import random, util
from game import Actions

from game import Agent


def genericsearch(problem, fringe):
    visited = set()
    # ( state, moves )
    startstate = problem.getStartState()
    fringe.push((startstate, 0))
    while not fringe.isEmpty():
        state, p = fringe.pop()
        if problem.isGoalState(state):
            return p

        if state not in visited:
            visited.add(state)
            for newstate, h in problem.getSuccessors(state):
                childstate = (newstate, p + h)
                fringe.push(childstate)


class SearchProblem:
    def getSuccessors(self, state):
        successors = []
        x, y = state
        for direction in [
            Directions.NORTH,
            Directions.SOUTH,
            Directions.EAST,
            Directions.WEST,
        ]:
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                successors.append(((nextx, nexty), 1))
        return successors


class ClosestFoodSearchProblem(SearchProblem):
    def __init__(self, gs):
        self.start = gs.getPacmanPosition()
        self.walls = gs.getWalls()
        self.foods = gs.getFood()

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        x, y = state
        return self.foods[x][y]


class MazeDistanceSearchProblem(SearchProblem):
    def __init__(self, gs, start, end):
        self.start = start
        self.end = end
        self.walls = gs.getWalls()

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return self.end == state


def closestfooddistance(gameState):
    problem = ClosestFoodSearchProblem(gameState)
    return genericsearch(problem, util.Queue())


def mazedistance(gameState, start, end):
    problem = MazeDistanceSearchProblem(gameState, start, end)

    def priorityf(x):
        pos, d = x
        return d

    d = genericsearch(problem, util.PriorityQueueWithFunction(priorityf))
    return 0 if d is None else d


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()

        "*** YOUR CODE HERE ***"
        foodleft = successorGameState.getNumFood()
        basescore = currentGameState.getScore() + successorGameState.getScore()
        foodscore = basescore * (1 / (foodleft ** 0.5)) if foodleft > 0 else basescore
        if currentGameState.getNumFood() == successorGameState.getNumFood():
            closestfood = closestfooddistance(successorGameState)
            foodscore -= closestfood

        # capsulesleft = len(successorGameState.getCapsules())
        # if capsulesleft > 0:
        # capscore = capsulesleft * basescore * .25
        # print capscore
        # foodscore -= capscore

        scaredghostdists = []
        ghostdists = []
        for g in newGhostStates:
            gdist = mazedistance(successorGameState, newPos, g.getPosition())
            if g.scaredTimer > gdist:
                scaredghostdists.append(gdist)
            else:
                ghostdists.append(gdist)

        avgghostdist = 0
        if len(ghostdists) > 0:
            avgghostdist = sum(ghostdists) / float(len(ghostdists))

        ghostscore = 0
        if avgghostdist > 0:
            ghostscore = basescore * (-2 / (avgghostdist ** 6))

        if sum(scaredghostdists) > 0:
            ghostscore += basescore * (1 / (sum(scaredghostdists) ** 0.5))

        gamescore = foodscore + ghostscore

        return gamescore
        # return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


def abminimax(agent, nextagent, state, depth, ms, mf, dispatch, a=-9999, b=9999):
    bestscore = ms
    bestmove = None

    for nextmove in state.getLegalActions(agent):
        nextstate = state.generateSuccessor(agent, nextmove)
        score, _ = dispatch(nextagent, nextstate, depth, a, b)
        bestscore, bestmove = mf((score, nextmove), (bestscore, bestmove))

        if agent == 0:
            # maximizer
            if bestscore > b:
                return bestscore, bestmove
            else:
                a = mf(a, bestscore)
        else:
            # minimizer
            if bestscore < a:
                return bestscore, bestmove
            else:
                b = mf(b, bestscore)

    return bestscore, bestmove


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        numagents = gameState.getNumAgents()
        searchdepth = numagents * self.depth

        def dispatch(agent, state, depth, action=None, *args):
            if state.isWin() or state.isLose() or depth < 1:
                return self.evaluationFunction(state), action
            else:
                nextagent = (agent + 1) % numagents
                if agent % numagents == 0:
                    return abminimax(
                        agent, nextagent, state, depth - 1, -9999, max, dispatch
                    )
                else:
                    return abminimax(
                        agent, nextagent, state, depth - 1, 9999, min, dispatch
                    )

        _, action = dispatch(0, gameState, searchdepth)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numagents = gameState.getNumAgents()
        searchdepth = numagents * self.depth

        def dispatch(agent, state, depth, a=-9999, b=9999, *args):
            if state.isWin() or state.isLose() or depth < 1:
                return self.evaluationFunction(state), None
            else:
                nextagent = (agent + 1) % numagents
                if agent % numagents == 0:
                    return abminimax(
                        agent, nextagent, state, depth - 1, -9999, max, dispatch, a, b
                    )
                else:
                    return abminimax(
                        agent, nextagent, state, depth - 1, 9999, min, dispatch, a, b
                    )

        _, action = dispatch(0, gameState, searchdepth)
        return action


def expectimax(agent, nextagent, state, depth, dispatch):
    scores = []
    actions = state.getLegalActions(agent)
    for nextmove in actions:
        nextstate = state.generateSuccessor(agent, nextmove)
        score, _ = dispatch(nextagent, nextstate, depth)
        scores.append(score)

    return sum(scores) / float(len(actions)), None


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        numagents = gameState.getNumAgents()
        searchdepth = numagents * self.depth

        def dispatch(agent, state, depth, a=-9999, b=9999, *args):
            if state.isWin() or state.isLose() or depth < 1:
                return self.evaluationFunction(state), None
            else:
                nextagent = (agent + 1) % numagents
                if agent % numagents == 0:
                    return abminimax(
                        agent, nextagent, state, depth - 1, -9999, max, dispatch, a, b
                    )
                else:
                    return expectimax(agent, nextagent, state, depth - 1, dispatch)

        _, action = dispatch(0, gameState, searchdepth)
        return action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanpos = currentGameState.getPacmanPosition()
    foodleft = currentGameState.getNumFood()
    basescore = currentGameState.getScore()
    foodscore = foodleft * 0.5
    if foodleft > 0:
        closestfood = closestfooddistance(currentGameState)
        foodscore -= closestfood * 0.1

    ghoststates = currentGameState.getGhostStates()
    scaredghostdists = []
    ghostdists = []
    for g in ghoststates:
        gdist = mazedistance(currentGameState, pacmanpos, g.getPosition())
        if g.scaredTimer > gdist:
            scaredghostdists.append(gdist)
        else:
            ghostdists.append(gdist)

    avgghostdist = 0
    if len(ghostdists) > 0:
        avgghostdist = sum(ghostdists) / float(len(ghostdists))

    ghostscore = 0
    if avgghostdist > 0:
        ghostscore = -2 / (avgghostdist ** 6)

    if sum(scaredghostdists) > 0:
        ghostscore += basescore * (1 / (sum(scaredghostdists) ** 0.5))

    gamescore = basescore + foodscore + ghostscore

    #    print gamescore

    return gamescore


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
