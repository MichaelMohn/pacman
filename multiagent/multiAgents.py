# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()
        closest = 0
        if newFood.asList():
            closest = min(util.manhattanDistance(newPos, food) for food in newFood.asList())
            score += 10.0 / closest



        activeGhosts = [
            util.manhattanDistance(newPos, ghost.getPosition()) for ghost, scared in zip(newGhostStates, newScaredTimes) if not scared]
        for d in activeGhosts:
            if d:
                score -= 5.0 / d
            else:
                return -10
            


        if action == Directions.STOP:
            score -= 2

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
  
        def minimax(state, depth, agentIndex):
            
            # base case
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()


            if agentIndex == 0:
                # Pacman
                bestVal = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    val = minimax(successor, depth, agentIndex + 1)
                    bestVal = max(bestVal, val)
                return bestVal
            else:
                # Ghost 
                bestVal = float("inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    # Next agent
                    if agentIndex == numAgents - 1:
                        val = minimax(successor, depth + 1, 0)
                    else:
                        val = minimax(successor, depth, agentIndex + 1)
                    bestVal = min(bestVal, val)
                return bestVal


        legalMoves = gameState.getLegalActions(0)
        maxScore = -10
        maxMove = None

        for action in legalMoves:
            #run minimax
            successor = gameState.generateSuccessor(0, action)
            val = minimax(successor, 0, 1)
            if val > maxScore:
                maxMove = action
                maxScore = val

        return maxMove

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(state, depth, agentIndex, alpha, beta):
            # Terminal check
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            if agentIndex == 0:
                # Pacman 
                value = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    val = alphabeta(successor, depth, agentIndex + 1, alpha, beta)
                    value = max(value, val)
                    if value > beta:
                        return value  # prune
                    alpha = max(alpha, value)
                return value
            else:
                # Ghosts 
                value = float("inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    if agentIndex == numAgents - 1:
                        val = alphabeta(successor, depth + 1, 0, alpha, beta)
                    else:
                        val = alphabeta(successor, depth, agentIndex + 1, alpha, beta)
                    value = min(value, val)
                    if value < alpha:
                        return value  # prune
                    beta = min(beta, value)
                return value

        # Root
        alpha = float("-inf")
        beta = float("inf")
        bestVal = float("-inf")
        bestAction = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            val = alphabeta(successor, 0, 1, alpha, beta)
            if val > bestVal:
                bestVal = val
                bestAction = action
            alpha = max(alpha, bestVal)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, agentIndex, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            actions= state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                return max(expectimax(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth) for action in actions)
            else:
                prob = 1.0/len(actions)
                return sum(
                    prob * expectimax(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth)
                    for action in actions
                )
        
        legal = gameState.getLegalActions(0)
        bestScore = float("-inf")
        best_action = Directions
        for action in legal:
            succ = gameState.generateSuccessor(0, action)
            score= expectimax(succ, 1, 0)
            if score > bestScore:
                bestScore, best_action = score, action
        return best_action
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    - Start from the game score
    - if the agent gets closer to the food, reward them. Penalize if they move away.
    - Heavily penalize being too close to an active ghost.
    - Reward getting near ghosts that can be eaten 
    """
    "*** YOUR CODE HERE ***"
    # Terminal states
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    score = float(currentGameState.getScore())

    foodCount = len(food)
    if foodCount:
        minFoodDist = min(manhattanDistance(pos, f) for f in food)
        food_closeness = 1.0 / max(1, minFoodDist)   # closer food is better
    else:
        minFoodDist = 1
        food_closeness = 0.0

    
    
    capCount = len(capsules)
    if capCount:
        minCapDist = min(manhattanDistance(pos, c) for c in capsules)
        cap_closeness = 1.0 / max(1, minCapDist)     # closer to capsules is good
    else:
        cap_closeness = 0.0

    
    
    danger_penalty = 0.0
    scared_reward = 0.0
    imminent_death = False

    for g in ghostStates:
        d = manhattanDistance(pos, g.getPosition())
        if g.scaredTimer > 0:
            
            
            scared_reward += 1.0 / max(1, d)
        else:
            if d <= 1:
                
                
                imminent_death = True
            elif d == 2:
                danger_penalty += 1.0
            elif d == 3:
                danger_penalty += 0.3

    if imminent_death:
        
        
        danger_penalty += 10.0

   
   
    try:
        mobility = len(currentGameState.getLegalActions(0))
    except Exception:
        mobility = 4  
        

    
    
    value = (
        1.0   * score           
        + 12.0 * food_closeness
        + 4.0  * cap_closeness
        - 4.0  * foodCount
        - 12.0 * capCount
        - 8.0  * danger_penalty
        + 8.0  * scared_reward
        + 0.5  * mobility
    )

    return value

    


better = betterEvaluationFunction
