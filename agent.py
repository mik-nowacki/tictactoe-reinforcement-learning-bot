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
INVALID_MOVE = -5


class Agent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(input_size=9, hidden_size=256, output_size=9)
        self.trainer = QTrainer(self.model, LR, self.gamma)
        self.model_name = "tic_tac_toe_model.pth"


    def save_model(self):
        torch.save(self.model.state_dict(), self.model_name)
        print(f"Model saved to {self.model_name}")
        

    def load_model(self):
        if os.path.exists(self.model_name):
            self.model.load_state_dict(torch.load(self.model_name))
            self.model.eval()
            print(f"Loaded model from {self.model_name}")
        else:
            print(f"No saved model found at {self.model_name}")


    def get_state(self, game: TicTacToe) -> list:
        state = game.board.copy()
        return state
    

    def get_action(self, state) -> tuple:
        # epsilon-greedy strategy
        self.epsilon = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * self.n_games / EPS_DECAY)  # exponential decay

        if random.choice(range(0, 100)) < self.epsilon:
            # random move (exploration)
            best_move_index = random.randint(0,8)
        else:
            # use a model to predict (exploitation)
            board_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
            Q = self.model(board_tensor)
            Q = Q.detach().numpy()
            best_move_index = np.argmax(Q)

        return tuple((best_move_index // 3, best_move_index % 3))
    

    def remember(self, old_state, action, reward, new_state, is_terminal):
        self.memory.append((old_state, action, reward, new_state, is_terminal))


    def train_short_memory(self, old_state, action, reward, new_state, is_terminal):
        action_index = action[0] * 3 + action[1]
        self.remember(old_state, action_index, reward, new_state, is_terminal)
        self.trainer.train_step(
            [old_state],
            [action_index],
            [reward],
            [new_state],
            [is_terminal])


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, is_terminals = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, is_terminals)



def random_player(available_moves):
    return random.choice(available_moves)



def train_ai():
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

            # Valid move check
            while True:
                action = agent.get_action(old_state)
                row, col = action
            
                # Check move validity
                if (row, col) in game.find_available_moves():
                    # Valid move
                    game.make_move(row, col)
                    new_state = agent.get_state(game)
                    break
                else:
                    # Invalid move punishment
                    agent.train_short_memory(old_state, action, INVALID_MOVE, old_state, False)
            
            # AI wins - end episode
            if game.check_winner() == AI:
                agent.train_short_memory(old_state, action, WIN, new_state, True)
                ai_wins+=1
                game_logs[episode] = 1
                break

            # Draw - end episode (after AI's episode)
            if len(game.find_available_moves()) == 0:
                agent.train_short_memory(old_state, action, DRAW, new_state, True)
                draws+=1
                game_logs[episode] = 0
                break

            r_row, r_col = random_player(game.find_available_moves())
            game.make_move(r_row, r_col)
            updated_state = agent.get_state(game)
            # Random player wins - end episode
            if game.check_winner() == RANDOM:
                agent.train_short_memory(old_state, action, LOSS, new_state, True)
                random_wins += 1
                game_logs[episode] = -1
                break

            # Draw - end episode (after player's move)
            if len(game.find_available_moves()) == 0:
                agent.train_short_memory(old_state, action, DRAW, new_state, True)
                draws+=1
                game_logs[episode] = 0
                break

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
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(game_logs), 50), win_logs, label="Wins")
    plt.plot(range(0, len(game_logs), 50), draw_logs, label="Draws")
    plt.plot(range(0, len(game_logs), 50), loss_logs, label="Losses")
    plt.title("AI Learning Progress Over Time")
    plt.xlabel("Episode")
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
    plt.ylabel("Result")
    plt.legend()
    plt.grid(True)
    plt.show()

    agent.save_model()
    return agent


def ai_starts(game: TicTacToe, ai: Agent):
    print("The game starts now! Type 'q q' to quit...")
    while True:
        # AI's move
        state = ai.get_state(game)
        # Valid move check
        while True:
            action = ai.get_action(state)
            row, col = action
        
            available_moves = game.find_available_moves()
            # Check move validity
            if (row, col) in available_moves:
                # Valid move
                game.make_move(row, col)
                break
            elif len(available_moves) == 0:
                # no moves left - terminal state
                break

        if len(game.find_available_moves()) == 0:
            print("It's a draw!")
            game.print_board()
            game.reset()
            continue
        
        # Player's move
        game.print_board()
        row, col = input("Make a move!").split()
        if row == 'q' or col == 'q':
            break
        game.make_move(int(row), int(col))

        if game.check_winner() is not None:
            print("We have a winner!")
            game.print_board()
            game.reset()
        elif len(game.find_available_moves()) == 0:
            print("It's a draw!")
            game.print_board()
            game.reset()


def player_starts(game: TicTacToe, ai: Agent):
    print("The game starts now! Type 'q q' to quit...")
    while True:
        # Player's move
        game.print_board()
        row, col = input("Make a move!").split()
        if row == 'q' or col == 'q':
            break
        game.make_move(int(row), int(col))

        # AI's move
        state = ai.get_state(game)
        # Valid move check
        while True:
            action = ai.get_action(state)
            row, col = action
        
            available_moves = game.find_available_moves()
            # Check move validity
            if (row, col) in available_moves:
                # Valid move
                game.make_move(row, col)
                break
            elif len(available_moves) == 0:
                # no moves left - terminal state
                break

        if game.check_winner() is not None:
            print("We have a winner!")
            game.print_board()
            game.reset()
        elif len(game.find_available_moves()) == 0:
            print("It's a draw!")
            game.print_board()
            game.reset()


def play(if_player_starts):
    game = TicTacToe()
    ai = Agent()
    ai.load_model()
    if not os.path.exists(ai.model_name):
        print("No trained model found, training new one...")
        ai = train_ai()
        ai.save_model()
    if if_player_starts.lower() == "yes":
        player_starts(game, ai)
    elif if_player_starts.lower() == "no":
        ai_starts(game, ai)
    else:
        print("Invalid input, bye!")
    
            

if __name__ == '__main__':
    play(input("Do you want to start? (yes/no): "))