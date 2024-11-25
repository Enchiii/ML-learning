import random
from collections import deque

from MlLib import utilities as util


class Agent:
    def __init__(self):
        memory_limit = 2**16
        self.memory = deque(maxlen=memory_limit)
        self.batch_size = 1024
        self.n_games = 0
        self.record = 0
        self.gamma = 0
        self.epsilon = 0
        self.model = None
        self.trainer = None

    def get_action(self, state):
        self.epsilon = 64 - self.n_games

        # robienie losowego ruchu co jakis czas (im epsilon jest mniejszy tym czestotliowsc bedzie mniejsza)
        if random.randint(0, 256) < self.epsilon:
            final_move = [0, 0, 0]
            move = random.randint(0, 2)
            final_move[move] = 1
            return final_move

        pred = self.model.predict(state)
        return util.argmax(pred)

    def update_memory(self, state, action, reward, new_state, game_over):
        self.memory.append([state, action, reward, new_state, game_over])

    def update_short_term_memory(self, state, action, reward, new_state, game_over):
        self.trainer.train_step(state, action, reward, new_state, game_over)

    def update_long_term_memory(self):
        sample = random.sample(self.memory, self.batch_size) if len(self.memory) > self.batch_size else self.memory

        # states, actions, rewards, new_states, game_overs = zip(*sample)
        self.trainer.train_step(zip(*sample))
