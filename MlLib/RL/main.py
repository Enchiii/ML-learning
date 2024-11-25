import sys

import pygame

from agent import Agent
from snake_game import SnakeGame

pygame.init()

WIDTH, HEIGHT = 400, 300
display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake")

clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 25)

if __name__ == "__main__":
    agent = Agent()
    game = SnakeGame(WIDTH, HEIGHT, display, clock, font)
    move_limit = game.w * game.h // SnakeGame.BLOCK_SIZE
    move_counter = 0

    # game loop
    while True:
        move_counter += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.K_q:
                pygame.quit()
                sys.exit()

        state = game.get_state()

        action = agent.get_action(state)  # model prediction

        game_over, score, reward = game.play_step(action)

        new_state = game.get_state()

        # train model
        agent.update_short_term_memory(state, action, reward, new_state, game_over)

        agent.update_memory(state, action, reward, new_state, game_over)

        if game_over or move_counter > move_limit:
            move_counter = 0
            agent.n_games += 1
            agent.update_long_term_memory()
            if score > agent.record:
                agent.record = score
                # agent.model.save()
            game.reset()

            print(f"Game {agent.n_games}, Score: {score}, Record {agent.record}")

            # TODO: plot
