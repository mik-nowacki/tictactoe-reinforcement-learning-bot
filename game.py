import numpy as np

PLAYER = 1
OPPONENT = -1
EMPTY = 0
SYMBOLS = {PLAYER: 'X', OPPONENT: 'O', EMPTY: '.'}

BOARD_SIZE = 3

class TicTacToe():
    def __init__(self):
        self.board = np.zeros(shape=(BOARD_SIZE,BOARD_SIZE), dtype=int)
        self.current_player = PLAYER  # "X" -> 1 (player), "O" -> -1 (bot), "Empty" -> 0


    def reset(self):
        self.board = np.zeros(shape=(BOARD_SIZE,BOARD_SIZE), dtype=int)
        self.current_player = PLAYER


    def make_move(self, row: int, col: int):
        self.board[row, col] = self.current_player
        self.current_player *= -1  # switch player
    

    def find_available_moves(self) -> list:
        return list(zip(*np.where(self.board == EMPTY)))


    def print_board(self):
        """Print the Tic-Tac-Toe board."""
        for row in self.board:
            print(" ".join(SYMBOLS[cell] for cell in row))
        print()


    def check_winner(self):
        for p in [PLAYER, OPPONENT]:
            for i in range(BOARD_SIZE):
                # Check rows and cols
                if np.all(self.board[i, :] == p) or np.all(self.board[:, i] == p):
                    return p
            # Check diagonals
            if np.all(np.diag(self.board) == p) or np.all(np.diag(np.fliplr(self.board)) == p):
                return p
        return None


    # def is_valid_move(self, row: int, col: int) -> bool:
    #     return (row, col) in self.find_available_moves()