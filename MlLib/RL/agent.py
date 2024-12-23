import random
import torch
from collections import deque
from model import *


class Agent:
    def __init__(self):
        memory_limit = 32_000
        self.memory = deque(maxlen=memory_limit)
        self.batch_size = 512
        self.n_games = 0
        self.record = 0
        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon = 0
        self.model = LinearQNet()
        self.model.load_state_dict(torch.load("./model/model.pth"))  # for continuing trained model
        self.model.train()  # for continuing trained model
        self.trainer = QTrainer(
            self.model,
            self.lr,
            self.gamma,
        )

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 0 - self.n_games  # 0 for continuing trained model
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def update_memory(self, state, action, reward, new_state, game_over):
        self.memory.append((state, action, reward, new_state, game_over))

    def update_short_term_memory(self, state, action, reward, new_state, game_over):
        self.trainer.train_step(state, action, reward, new_state, game_over)

    def update_long_term_memory(self):
        sample = random.sample(self.memory, self.batch_size) if len(self.memory) > self.batch_size else self.memory

        states, actions, rewards, new_states, game_overs = zip(*sample)
        self.trainer.train_step(states, actions, rewards, new_states, game_overs)
