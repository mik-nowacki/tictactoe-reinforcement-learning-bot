import numpy as np
import random

# Constants
PLAYER = 1
OPPONENT = -1
EMPTY = 0
DRAW = 0
BOARD_SIZE = 3

class Node:
    def __init__(self, state, parent=None):
        self.state = state  # Board state (3x3 numpy array)
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0  # Total reward from this node
        self.untried_actions = self.get_legal_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def select_child(self, exploration_weight=1.414):
        """Select a child node using the UCT formula."""
        log_parent_visits = np.log(self.visits)
        best_child = None
        best_score = -float('inf')

        for child in self.children:
            exploitation = child.value / child.visits
            exploration = np.sqrt(log_parent_visits / child.visits)
            uct_score = exploitation + exploration_weight * exploration

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child

    def expand(self):
        """Expand a node by adding a new child."""
        action = self.untried_actions.pop()
        new_state = self.state.copy()
        new_state[action] = PLAYER if self.state.sum() == 0 else OPPONENT
        new_node = Node(new_state, self)
        self.children.append(new_node)
        return new_node

    def get_legal_actions(self):
        """Get all legal actions (empty cells) on the board."""
        return list(zip(*np.where(self.state == EMPTY)))

    def is_terminal(self):
        """Check if the node is a terminal state (win, lose, or draw)."""
        return self.check_winner() is not None or len(self.get_legal_actions()) == 0

    def check_winner(self):
        """Check if the current state has a winner."""
        for player in [PLAYER, OPPONENT]:
            # Check rows and columns
            for i in range(BOARD_SIZE):
                if np.all(self.state[i, :] == player) or np.all(self.state[:, i] == player):
                    return player
            # Check diagonals
            if np.all(np.diag(self.state) == player) or np.all(np.diag(np.fliplr(self.state)) == player):
                return player
        return None


class MCTS:
    def __init__(self, iterations=1000):
        self.iterations = iterations

    def search(self, initial_state):
        """Perform MCTS from the given initial state."""
        self.root = Node(initial_state)

        for _ in range(self.iterations):
            node = self.select(self.root)
            winner = self.simulate(node)
            self.backpropagate(node, winner)

        # Choose the best action based on the most visited child
        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.state

    def select(self, node):
        """Select a node to expand."""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.select_child()
        return node

    def simulate(self, node):
        """Simulate a random game from the given node."""
        state = node.state.copy()
        current_player = PLAYER if state.sum() == 0 else OPPONENT

        while True:
            legal_actions = list(zip(*np.where(state == EMPTY)))
            if not legal_actions:
                return DRAW  # Draw

            action = random.choice(legal_actions)
            state[action] = current_player

            winner = Node(state).check_winner()
            if winner is not None:
                return winner

            current_player = OPPONENT if current_player == PLAYER else PLAYER

    def backpropagate(self, node, winner):
        """Backpropagate the result of a simulation."""
        while node is not None:
            node.visits += 1
            if winner == PLAYER:
                node.value += 1
            elif winner == OPPONENT:
                node.value -= 1
            node = node.parent
