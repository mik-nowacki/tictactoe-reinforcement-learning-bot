import numpy as np
from mcts import Node, MCTS, PLAYER, OPPONENT, EMPTY, BOARD_SIZE



def print_board(board):
    """Print the Tic-Tac-Toe board."""
    symbols = {PLAYER: 'X', OPPONENT: 'O', EMPTY: '.'}
    for row in board:
        print(" ".join(symbols[cell] for cell in row))
    print()


def play_game():
    """Play a game of Tic-Tac-Toe against the MCTS algorithm."""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    mcts = MCTS(iterations=1000)

    while True:
        # Player's turn
        print("Your turn (X):")
        row, col = map(int, input("Enter row and column (0-2): ").split())
        if board[row, col] != EMPTY:
            print("Invalid move! Try again.")
            continue
        board[row, col] = PLAYER
        print_board(board)

        # Check if player wins
        winner = Node(board).check_winner()
        if winner is not None:
            print("You win!")
            break
        if len(Node(board).get_legal_actions()) == 0:
            print("It's a draw!")
            break

        # MCTS's turn
        print("MCTS's turn (O):")
        board = mcts.search(board)
        print_board(board)

        # Check if MCTS wins
        winner = Node(board).check_winner()
        if winner is not None:
            print("MCTS wins!")
            break
        if len(Node(board).get_legal_actions()) == 0:
            print("It's a draw!")
            break


if __name__ == "__main__":
    play_game()