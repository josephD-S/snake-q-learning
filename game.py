import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math
import time

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

BLOCK_SIZE = 40
SPEED = 100

class SnakeGameAI:
    def __init__(self, w=200, h=200):
        # Constants
        self.w = w
        self.h = h
        self.max_x = int(self.w//BLOCK_SIZE)
        self.max_y = int(self.h//BLOCK_SIZE)

        # Params
        self.c = 3 # Distance change reward
        self.local_size = 5 # Size of local map (y, x)
        self.local_half = self.local_size // 2 if self.local_size % 2 == 1 else self.local_size // 2 - 1

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.max_dist = math.hypot((self.w // BLOCK_SIZE)-1, (self.h // BLOCK_SIZE)-1)
        
        # Begin snake
        self.reset()

    def get_height(self):
        return self.local_size
    
    def get_width(self):
        return self.local_size

    def retrieve_local_map(self):
        local_map = np.zeros((self.local_size, self.local_size), dtype=np.int8)

        # Draw head
        local_map[self.local_half, self.local_half] = 1
        
        # Get head coordinates
        head_x, head_y = int(self.head[0]//BLOCK_SIZE), int(self.head[1]//BLOCK_SIZE)

        # Draw body
        for snake_body in self.snake[1:]:
            # Get body x and y
            x, y = int(snake_body[0]//BLOCK_SIZE), int(snake_body[1]//BLOCK_SIZE)

            # Get distance from head
            dx = head_x - x 
            dy = head_y - y

            # Check if outside local area
            if abs(dx) <= self.local_half and abs(dy) <= self.local_half:
                # Get local body indices
                local_x = self.local_half - dx 
                local_y = self.local_half - dy

                # Draw body
                local_map[local_y, local_x] = 1

        # Draw walls
        dx_wall_right = self.max_x - head_x
        dy_wall_bottom = self.max_y - head_y 

        # Draw x walls
        if abs(dx_wall_right) <= self.local_half:
            if abs(dy_wall_bottom) <= self.local_half:
                # Bottom right 
                local_map[:self.local_half+dy_wall_bottom, self.local_half+dx_wall_right] = 1
                local_map[self.local_half+dy_wall_bottom, :self.local_half+dx_wall_right+1] = 1

            elif head_y <= self.local_half: 
                # Top right
                local_map[self.local_half-head_y-1:, self.local_half+dx_wall_right] = 1
                local_map[self.local_half-head_y-1, :self.local_half+dx_wall_right] = 1

            else:
                # Only right 
                local_map[:, self.local_half+dx_wall_right:] = 1

        if head_x <= self.local_half:
            if abs(dy_wall_bottom) <= self.local_half:
                # Bottom left
                local_map[:self.local_half+dy_wall_bottom, self.local_half-head_x-1] = 1
                local_map[self.local_half+dy_wall_bottom, self.local_half-head_x-1:] = 1
                
            elif head_y <= self.local_half:
                # Top left
                local_map[self.local_half-head_y-1:, self.local_half-head_x-1] = 1
                local_map[self.local_half-head_y-1, self.local_half-head_x-1:] = 1

            else:
                # Only left
                local_map[:, self.local_half-head_x-1:] = 1

        if abs(dy_wall_bottom) <= self.local_half and not (
            abs(dx_wall_right) <= self.local_half or
            head_x <= self.local_half
            ):
            # Draw walls below
            local_map[self.local_half+dy_wall_bottom:, :] = 1
        
        if head_y <= self.local_half and not (
            abs(dx_wall_right) <= self.local_half or
            head_x <= self.local_half
            ):
            # Draw walls above
            local_map[:self.local_half-head_y-1, :] = 1

        # Draw food
        x_food, y_food = int(self.food[0]//BLOCK_SIZE), int(self.food[1]//BLOCK_SIZE)
        dx_food = head_x - x_food
        dy_food = head_y - y_food

        if abs(dx_food) <= self.local_half and abs(dy_food) <= self.local_half:
            local_x = self.local_half - dx_food
            local_y = self.local_half - dy_food 

            local_map[local_y, local_x] = -1

        return local_map

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(BLOCK_SIZE*2, BLOCK_SIZE*2)
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
        if self.is_collision() or self.frame_iteration > 500*len(self.snake):
            game_over = True
            reward = -50
            return reward, game_over, self.score
            
        # 4. place new food or just move
        ate = False
        if self.head == self.food:
            self.score += 1
            reward = 100
            self.without_food_frame = 0
            ate = True
            self._place_food()
        else:
            self.snake.pop()
            reward = -2
        
        new_dist = math.sqrt((self.head[0]/BLOCK_SIZE - self.food[0]/BLOCK_SIZE)**2 +
                  (self.head[1]/BLOCK_SIZE - self.food[1]/BLOCK_SIZE)**2)
        

        dist_change = (self.prev_dist - new_dist) / self.max_dist
        #if not ate:
        #    reward += dist_change * self.c
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
            #pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            
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
