# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            oldValues = self.values.copy()
            newValues = util.Counter()

            for s in self.mdp.getStates():
                if self.mdp.isTerminal(s):
                    newValues[s] = 0.0
                    continue

                actions = self.mdp.getPossibleActions(s)
                best_q = max(
                    sum(p * (self.mdp.getReward(s, a, sp) + self.discount * oldValues[sp])
                        for sp, p in self.mdp.getTransitionStatesAndProbs(s, a))
                    for a in actions
                )
                newValues[s] = best_q

            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0.0
        for sp, p in self.mdp.getTransitionStatesAndProbs(state, action):
            r = self.mdp.getReward(state, action, sp)
            q += p * (r + self.discount * self.values[sp])
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        # util.Counter has argMax, but we can do it directly:
        best_action = max(actions, key=lambda a: self.computeQValueFromValues(state, a))
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()

        for i in range(self.iterations):
            mod = i % len(states)
            state = states[mod]

            if self.mdp.isTerminal(state):
                continue

            values = []
            for action in self.mdp.getPossibleActions(state):
                q = 0
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    reward = self.mdp.getReward(state, action, next_state)
                    disc = self.discount * self.values[next_state]
                    q = q + prob * (reward + disc)
                
                values.append(q)

            if values:
                self.values[state] = max(values)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()

        pred = {}
        for s in states:
            pred[s] = set()

        for s in states:
            if self.mdp.isTerminal(s):
                continue

            for act in self.mdp.getPossibleActions(s):
                for next, p in self.mdp.getTransitionStatesAndProbs(s, act):
                    
                    if p > 0:
                        pred[next].add(s)

        prio = util.PriorityQueue()
        for s in states:
            if self.mdp.isTerminal(s):
                continue

            vals = [self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)]
            maxq = max(vals) if vals else 0
            prio.update(s, -abs(self.values[s] - maxq))


        for i in range(self.iterations):
            if prio.isEmpty():
                break

            curstate = prio.pop()
            if not self.mdp.isTerminal(curstate):
                vals = [self.computeQValueFromValues(curstate, a) for a in self.mdp.getPossibleActions(curstate)]
                self.values[curstate] = max(vals) if vals else 0

            for p in pred[curstate]:
                if self.mdp.isTerminal(p):
                    continue
                vals = [self.computeQValueFromValues(p, a) for a in self.mdp.getPossibleActions(p)]
                maxq = max(vals) if vals else 0
                if abs(self.values[p] - maxq) > self.theta:
                    prio.update(p, -abs(self.values[p] - maxq))

