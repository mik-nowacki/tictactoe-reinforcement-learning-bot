import random
import numpy as np
from collections import deque, Counter
import os
from game import TicTacToe, SYMBOLS, PLAYER, OPPONENT, EMPTY
import torch
from model import Linear_QNet, QTrainer
import matplotlib.pyplot as plt
import math

MAX_MEMORY = 10_000
BATCH_SIZE = 100
LR = 0.001  # learning rate
N_EPISODES = 10_000
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000

# rewards
WIN = 10
LOSS = -10 
DRAW = 5
ONGOING = 0


class Agent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(input_size=9, hidden_size=256, output_size=9)
        self.trainer = QTrainer(self.model, LR, self.gamma)


    def get_state(self, game: TicTacToe) -> list:
        state = game.board.copy()
        return state


    def find_available_moves(self, state) -> list:
        # Ensure 2D board format
        board = state.reshape(3, 3) if state.size == 9 else state
        return list(zip(*np.where(board == EMPTY)))
    

    def get_action(self, state) -> tuple:
        # epsilon-greedy strategy
        # current_board, available_moves = state
        self.epsilon = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * self.n_games / EPS_DECAY)
        # self.epsilon = 100 * (0.95 ** self.n_games)  # 5% decay per episode
        available_moves = self.find_available_moves(state)
        if random.choice(range(0, 100)) < self.epsilon:
            # random move (exploration)
            return random.choice(available_moves)

        # use a model to predict (exploitation)
        board_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
        Q = self.model(board_tensor)
        Q = Q.detach().numpy()
        for i in range(9):
            if (i // 3, i % 3) not in available_moves:
                Q[i] = -float('inf')

        best_move_index = np.argmax(Q)

        return tuple((best_move_index // 3, best_move_index % 3))
    

    def remember(self, old_state, action, reward, new_state, is_terminal):
        self.memory.append((old_state, action, reward, new_state, is_terminal))


    def train_short_memory(self, old_state, action, reward, new_state, is_terminal):
        action_index = action[0] * 3 + action[1]
        # Wrap available moves in a list to match batch format
        available_moves = [self.find_available_moves(old_state)]  # Note extra []
        self.trainer.train_step(
            [old_state],  # Wrap state in list
            available_moves,
            [action_index],  # Wrap action in list
            [reward],  # Wrap reward in list
            [new_state],  # Wrap new_state in list
            [is_terminal]  # Wrap terminal flag in list
        )
        self.remember(old_state, action_index, reward, new_state, is_terminal)

        

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, is_terminals = zip(*sample)
        available_moves_per_state = [self.find_available_moves(state) for state in states]
        self.trainer.train_step(states, available_moves_per_state, actions, rewards, next_states, is_terminals)



def random_player(available_moves):
    return random.choice(available_moves)



def train():
    game = TicTacToe()
    agent = Agent()
    AI = PLAYER
    RANDOM = OPPONENT

    ai_wins = 0
    random_wins = 0
    draws = 0

    game_logs = np.zeros(N_EPISODES)

    for episode in range(N_EPISODES):
        game.reset()
        while True:
            # AI Turn
            old_state = agent.get_state(game)
            action = agent.get_action(old_state)
            game.make_move(row=action[0], col=action[1])
            new_state = agent.get_state(game)

            # AI wins - end episode
            if game.check_winner() == AI:
                agent.remember(old_state, action, WIN, new_state, True)
                agent.train_short_memory(old_state, action, WIN, new_state, True)
                ai_wins+=1
                game_logs[episode] = 1
                break

            # Draw - end episode (after AI's episode)
            if len(agent.find_available_moves(new_state)) == 0:
                agent.remember(old_state, action, DRAW, new_state, True)
                agent.train_short_memory(old_state, action, DRAW, new_state, True)
                draws+=1
                game_logs[episode] = 0
                break

            r_row, r_col = random_player(agent.find_available_moves(new_state))
            game.make_move(r_row, r_col)
            updated_state = agent.get_state(game)
            # Random player wins - end episode
            if game.check_winner() == RANDOM:
                agent.remember(old_state, action, LOSS, new_state, True)
                random_wins += 1
                game_logs[episode] = -1
                break

            # Draw - end episode (after player's move)
            if len(agent.find_available_moves(updated_state)) == 0:
                agent.remember(old_state, action, DRAW, new_state, True)
                agent.train_short_memory(old_state, action, DRAW, new_state, True)
                draws+=1
                game_logs[episode] = 0
                break

            # agent.remember(old_state, action, ONGOING, updated_state, False)
            # agent.train_short_memory(old_state, action, ONGOING, updated_state, False)
            agent.train_short_memory(old_state, action, ONGOING, new_state, False)


        agent.n_games+=1
        agent.train_long_memory()
    
    print("AI wins: ", ai_wins)
    print("Random_player wins: ", random_wins)
    print("Draws: ", draws)

    # Count alltogether
    win_logs = []
    draw_logs = []
    loss_logs = []
    for i in range(50, len(game_logs)+1, 50):
        counter = Counter(game_logs[:i])
        win_logs.append(counter[1])
        draw_logs.append(counter[0])
        loss_logs.append(counter[-1])
        
    # Plotting the results
    # plt.figure(figsize=(10, 6))
    # plt.scatter(range(len(game_logs)),game_logs, s=50, alpha=0.5)
    # plt.title("AI Learning Progress Over Time")
    # plt.xlabel("Episode")
    # plt.yticks(ticks=[-1, 0, 1], labels=['Loss', 'Draw', 'Win'])
    # plt.ylabel("Result")
    # plt.grid(True)
    # plt.show()

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(game_logs), 50), win_logs, label="Wins")
    plt.plot(range(0, len(game_logs), 50), draw_logs, label="Draws")
    plt.plot(range(0, len(game_logs), 50), loss_logs, label="Losses")
    plt.title("AI Learning Progress Over Time")
    plt.xlabel("Episode")
    # plt.yticks(ticks=[-1, 0, 1], labels=['Loss', 'Draw', 'Win'])
    plt.ylabel("Result")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Count per 50 samples
    win_logs = []
    draw_logs = []
    loss_logs = []
    for i in range(0, len(game_logs), 50):
        counter = Counter(game_logs[i:i+50])
        win_logs.append(counter[1])
        draw_logs.append(counter[0])
        loss_logs.append(counter[-1])

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(game_logs), 50), win_logs, label="Wins")
    plt.plot(range(0, len(game_logs), 50), draw_logs, label="Draws")
    plt.plot(range(0, len(game_logs), 50), loss_logs, label="Losses")
    plt.title("AI Learning Progress Over Time")
    plt.xlabel("Episode")
    # plt.yticks(ticks=[-1, 0, 1], labels=['Loss', 'Draw', 'Win'])
    plt.ylabel("Result")
    plt.legend()
    plt.grid(True)
    plt.show()



# def play():
#     game = TicTacToe()
#     print("The game starts now! Type 'q q' to quit...")
#     game.print_board()
#     while True:
#         row, col = input("Make a move!").split()
#         if row == 'q' or col == 'q':
#             break
#         game.make_move(int(row), int(col))
#         game.print_board()

#         if winner := game.check_winner() is not None:
#             print(f"{SYMBOLS[winner]} wins!")
#             break
#         elif len(game.find_available_moves()) == 0:
#             print("It's a draw!")
#             break


if __name__ == '__main__':
        train()
        # play()