import sys

import pygame
import numpy as np

from agent import Agent
from snake_game import SnakeGame
from helper import plot

pygame.init()

WIDTH = 640  # Must be divisible by 20
HEIGHT = 480  # Must be divisible by 20
display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake")

clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 25)

if __name__ == "__main__":
    agent = Agent()
    game = SnakeGame(WIDTH, HEIGHT, display, clock, font)
    scores = []
    mean_scores = []

    # game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.K_q:
                pygame.quit()
                sys.exit()

        state = game.get_state()

        action = agent.get_action(state)  # model prediction

        action = list(action)

        game_over, score, reward = game.play_step(action)

        new_state = game.get_state()

        # train model
        agent.update_short_term_memory(state, action, reward, new_state, game_over)

        agent.update_memory(state, action, reward, new_state, game_over)

        if game_over:
            agent.n_games += 1
            scores.append(score)
            mean_scores.append(np.mean(scores))
            agent.update_long_term_memory()
            if score > agent.record:
                agent.record = score
                agent.model.save()
            game.reset()

            print(f"Game {agent.n_games}, Score: {score}, Record {agent.record}, Mean {np.mean(mean_scores)}")

            plot(scores, mean_scores)
