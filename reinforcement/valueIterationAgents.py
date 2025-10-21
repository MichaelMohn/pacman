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
            # Select one state in cyclic order
            state = states[i % len(states)]

            # Skip terminal states
            if self.mdp.isTerminal(state):
                continue

            # Compute the best action value
            action_values = []
            for action in self.mdp.getPossibleActions(state):
                q_value = 0
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    reward = self.mdp.getReward(state, action, next_state)
                    q_value += prob * (reward + self.discount * self.values[next_state])
                action_values.append(q_value)

            # Update value for the current state
            if action_values:
                self.values[state] = max(action_values)

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
        predecessors = {}

        for s in states:
            predecessors[s] = set()

        for s in states:
            if self.mdp.isTerminal(s):
                continue
            for a in self.mdp.getPossibleActions(s):
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(s, a):
                    if prob > 0:
                        predecessors[next_state].add(s)

        pq = util.PriorityQueue()
        theta = 1e-5

        for s in states:
            if self.mdp.isTerminal(s):
                continue
            q_values = [self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)]
            best_q = max(q_values) if q_values else 0
            diff = abs(self.values[s] - best_q)
            pq.update(s, -diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                break

            s = pq.pop()
            if not self.mdp.isTerminal(s):
                q_values = [self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)]
                self.values[s] = max(q_values) if q_values else 0

            for p in predecessors[s]:
                if self.mdp.isTerminal(p):
                    continue
                q_values = [self.computeQValueFromValues(p, a) for a in self.mdp.getPossibleActions(p)]
                best_q = max(q_values) if q_values else 0
                diff = abs(self.values[p] - best_q)
                if diff > self.theta:
                    pq.update(p, -diff)

