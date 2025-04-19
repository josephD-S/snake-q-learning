import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 1000

class SnakeGameAI:
    def __init__(self, w=640, h=640):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.max_dist = math.hypot((self.w // BLOCK_SIZE)-1, (self.h // BLOCK_SIZE)-1)
        self.c = 5
        self.reset()
        
    def retrieve_map(self):
        map = np.zeros((2, self.h//BLOCK_SIZE, self.w//BLOCK_SIZE), dtype='I')

        snake_indices = []
        for point in self.snake:
            snake_indices.append((int(point[0]//BLOCK_SIZE), int(point[1]//BLOCK_SIZE)))

        for pos in snake_indices:
            map[0, pos[1]-1, pos[0]-1] = 1

        map[1, int(self.food[0]//BLOCK_SIZE)-1, int(self.food[1]//BLOCK_SIZE)-1] = 1 # Food
        map[1, int(self.head[0]//BLOCK_SIZE)-1, int(self.head[1]//BLOCK_SIZE)-1] = 1 # Head

        return map

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.without_food_frame = 0
        self.prev_dist = math.hypot(
            (self.head.x/BLOCK_SIZE - self.food.x/BLOCK_SIZE),
            (self.head.y/BLOCK_SIZE - self.food.y/BLOCK_SIZE)
        )
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1
        self.without_food_frame += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 200*len(self.snake):
            game_over = True
            reward = -1
            return reward, game_over, self.score
            
        # 4. place new food or just move
        ate = False
        if self.head == self.food:
            self.score += 1
            reward = 1
            self.without_food_frame = 0
            ate = True
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.10
        
        new_dist = math.sqrt((self.head[0]/BLOCK_SIZE - self.food[0]/BLOCK_SIZE)**2 +
                  (self.head[1]/BLOCK_SIZE - self.food[1]/BLOCK_SIZE)**2)
        

        dist_change = (self.prev_dist - new_dist) / self.max_dist
        if not ate:
            reward += dist_change * self.c
        self.prev_dist = new_dist

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_direction = clock_wise[next_idx] # right turn r->d d->l
        else:
            next_idx = (idx - 1) % 4
            new_direction = clock_wise[next_idx] # left turn

        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
