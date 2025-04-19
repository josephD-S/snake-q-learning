import torch 
import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Randomness param
        self.gamma = 0.95 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # pops as len approaches MAX_MEMORY
        self.model = Linear_QNet(1024, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        map = game.retrieve_map()
        map = map.flatten()

        direction = [
            game.direction == Direction.LEFT,
            game.direction == Direction.RIGHT,
            game.direction == Direction.UP,
            game.direction == Direction.DOWN
        ]

        direction = np.array(direction, dtype=int)

        state = np.concatenate((map, direction))

        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # Append complete state. popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Returns list of tuples
        else:
            mini_sample = self.memory 

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        avg_q_value = self.trainer.train_step(states, actions, rewards, next_states, dones)

        return avg_q_value

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # In the beginning, we want random moves to do more exploration than exploitation, i.e. exploration / exploitation tradeoff
        self.epsilon = 500 - self.n_games
        final_move = [0, 0, 0]
        # Pick a random move
        if random.randint(0, 500) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Use a predicted move from model
            prediction = self.model(state)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_q_values = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get current state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.n_games += 1
            max_q_value = agent.train_long_memory()

            if score > record:
                record = score 
                agent.model.save()

            print('Game ', agent.n_games, 'Score ', score, 'Record: ', record)

            plot_scores.append(score)
            total_score += score 
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot_q_values.append(max_q_value)
            plot(plot_scores, plot_mean_scores, plot_q_values)



if __name__ == "__main__":
    train()


