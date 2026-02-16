# Tic-Tac-Toe Deep Q-Learning Bot

A Deep Reinforcement Learning project that implements a **Deep Q-Network (DQN)** to train an agent to play Tic-Tac-Toe. The agent evolves from making random moves to developing an optimal strategy through a reward-based feedback loop.

## Project Architecture

This bot uses a **Neural Network** to approximate the Q-value function. This allows the agent to generalize its learning across the board's state space more efficiently.

### Core Components:

- **Linear Q-Net:** A 3-layer feed-forward neural network (9 inputs → 256 hidden → 256 hidden → 9 outputs).
- **Experience Replay (Long Memory):** Stores up to 10,000 transitions in a `deque` buffer. Training on random batches from this memory prevents the model from "forgetting" old scenarios and stabilizes the learning process.
- **Short Memory:** Provides immediate reinforcement by training the model on the very last move made.
- **Epsilon-Greedy Strategy:** \* **Exploration:** Initially, the bot makes random moves to "discover" the game.
  - **Exploitation:** Over time, it uses an exponential decay function ($e^{-x}$) to stop guessing and start using its trained model to win.

---

## Reward System

The agent is trained using a reward structure to shape its behavior:

| Event            | Reward |
| :--------------- | :----- |
| **Win**          | `+10`  |
| **Loss**         | `-10`  |
| **Draw**         | `+5`   |
| **Correct Move** | `0`    |
| **Invalid Move** | `-5`   |

---

## Performance

The bot is trained against a "Random Choice" opponent baseline.
It achieves ~**95% win rate** within 1,500 episodes.
