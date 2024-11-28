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

        # Perform value iteration for self.iterations times
        for i in range(self.iterations):
            states = self.mdp.getStates()
            temp_counter = util.Counter()  # Temporary counter to store updated values

            # Loop over all states
            for state in states:
                # Call computeActionFromValues to get the best action's Q-value
                best_action = self.computeActionFromValues(state)

                # Possibility of best_action being None, so account for this
                if best_action is not None:
                    temp_counter[state] = self.computeQValueFromValues(state, best_action)
                else:
                    temp_counter[state] = 0

            # Update values
            self.values = temp_counter

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

        qVal = 0
        actionPairs = self.mdp.getTransitionStatesAndProbs(state, action)

        for next_state, prob in actionPairs:
            reward = self.mdp.getReward(state, action, next_state)
            qVal += prob * (reward + self.discount * self.values[next_state])
        return qVal


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        maxVal = float('-inf')
        bestAction = None
        actions = self.mdp.getPossibleActions(state)

        # Return none if there are no legal actions
        if not actions:
            return None
        
        # Go through each action, find max action
        for action in actions:
            tmp = self.computeQValueFromValues(state, action)
            if tmp > maxVal:
                maxVal = tmp
                bestAction = action
        
        return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
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
        "*** YOUR CODE HERE ***"

        # I made this helper function, as this block of code was repeated in my logic
        def GetBestActionValue(state):
            best_action = self.computeActionFromValues(state)
            if best_action:
                return self.computeQValueFromValues(state, best_action)
            else:
                return 0

        # Initialize Priority Queue using util.PriorityQueue() and predecessor tracking
        pq = util.PriorityQueue()
        predecessors = {state: set() for state in self.mdp.getStates()}

        # Track state dependencies (aka the predecessors)
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        predecessors[next_state].add(state)

        # Initialize priority queue with state value differences
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                best_action_value = GetBestActionValue(state)
                diff = abs(best_action_value - self.values[state])
                pq.update(state, -diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                break

            # Pop state from queue, update its value, and push neighbours back into the queue
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                best_action_value = GetBestActionValue(state)
                self.values[state] = best_action_value

                # Update predecessors values and push to the priority queue
                for pred_state in predecessors[state]:
                    if not self.mdp.isTerminal(pred_state):
                        best_action_value = GetBestActionValue(pred_state)
                        diff = abs(best_action_value - self.values[pred_state])
                        if diff > self.theta:
                            pq.update(pred_state, -diff)