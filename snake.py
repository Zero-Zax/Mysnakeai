import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pygame
import random
import os

os.environ["SDL_VIDEODRIVER"] = "x11"

class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.buffer = []
        self.max_size = max_size
        self.len = 0

    def add(self, experience):
        if self.len < self.max_size:
            self.buffer.append(experience)
            self.len += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def sample(self, batch_size):
        index = np.random.randint(0, len(self.buffer), size=batch_size)
        return [self.buffer[i] for i in index]

def create_model(input_size, output_size):
    model = Sequential([
        Dense(256, input_dim=input_size, activation='relu'),
        Dense(150, activation='relu'),
        Dense(output_size, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

class SnakeGameAI:
    def __init__(self):
        pygame.init()
        self.screen_width = 850
        self.screen_height = 750
        self.game_screen = pygame.display.set_mode((850, 750))
        pygame.display.set_caption("Snake Game AI")
        self.clock = pygame.time.Clock()
        self.snake_size = 50
        self.reset()

    def reset(self):
        self.snake_pos = [400, 400]
        self.snake_body = [[400, 400], [350, 400], [300, 400]]
        self.direction = 'RIGHT'
        self.food_pos = self._place_food()
        self.score = 0
        self.frame_iteration = 0
        self.moves_since_last_food = 0
        return self.get_state()

    def _place_food(self):
        return [random.randrange(1, (self.screen_width // self.snake_size)) * self.snake_size, 
                random.randrange(1, (self.screen_height // self.snake_size)) * self.snake_size]

    def _update_ui(self):
        self.game_screen.fill((0, 0, 0))
        for part in self.snake_body:
            pygame.draw.rect(self.game_screen, (0, 255, 0), pygame.Rect(part[0], part[1], self.snake_size, self.snake_size))
        pygame.draw.rect(self.game_screen, (255, 0, 0), pygame.Rect(self.food_pos[0], self.food_pos[1], self.snake_size, self.snake_size))
        pygame.display.update()

    # Define movement and action logic
    def _move(self, action):
        # Simplify action logic: 0 - straight, 1 - right, 2 - left
        if action == 1:  # Right turn
            if self.direction == 'UP':
                self.direction = 'RIGHT'
            elif self.direction == 'RIGHT':
                self.direction = 'DOWN'
            elif self.direction == 'DOWN':
                self.direction = 'LEFT'
            elif self.direction == 'LEFT':
                self.direction = 'UP'
        elif action == 2:  # Left turn
            if self.direction == 'UP':
                self.direction = 'LEFT'
            elif self.direction == 'LEFT':
                self.direction = 'DOWN'
            elif self.direction == 'DOWN':
                self.direction = 'RIGHT'
            elif self.direction == 'RIGHT':
                self.direction = 'UP'

        if self.direction == 'UP':
            self.snake_pos[1] -= self.snake_size
        elif self.direction == 'DOWN':
            self.snake_pos[1] += self.snake_size
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= self.snake_size
        elif self.direction == 'RIGHT':
            self.snake_pos[0] += self.snake_size

        # Move the snake
        self.snake_body.insert(0, list(self.snake_pos))
        # Check if food is eaten
        if not self.snake_pos == self.food_pos:
            self.snake_body.pop()  # Remove the last piece if no food is eaten
            self.moves_since_last_food += 1  # Increment the move counter
        else:
            self.score += 1  # Increase score for eating food
            self.food_pos = self._place_food()  # Place new food
            self.moves_since_last_food = 0


    # Check for collisions with walls or self
    def _is_collision(self, pt=None):
        
        if pt is None:
            pt = self.snake_pos
        # Check if the snake head is out of bounds
        if pt[0] < 0 or pt[0] >= self.screen_width:
            return True
        if pt[1] < 0 or pt[1] >= self.screen_height:
            return True
        # Check if the snake has collided with itself
        if pt in self.snake_body[1:]:
            return True
        print(pt[0], pt[1])
        return False

    def not_moving(self):
        # Check if the snake has not moved by comparing the first and second segment of the snake's body
        return self.snake_body[0] == self.snake_body[1]
    
    # Play a single step of the game
    def play_step(self, action):
        next_state = self.get_state()
        self.frame_iteration += 1
        self._move(action)
        
        game_over = self._is_collision() or self.moves_since_last_food >= 200
        if self.not_moving():
            game_over = True  # Check if the game should end
        reward = -10 if game_over else 0
        
        reward -= 1
        self._update_ui()
        self.clock.tick(1000)
        return next_state, reward, game_over, self.score

    # Get the state of the game
    def get_state(self):
        # Define the state based on the direction of the snake, food location, and danger
        head = self.snake_pos
        point_l = [head[0] - self.snake_size, head[1]]
        point_r = [head[0] + self.snake_size, head[1]]
        point_u = [head[0], head[1] - self.snake_size]
        point_d = [head[0], head[1] + self.snake_size]

        dir_l = self.direction == 'LEFT'
        dir_r = self.direction == 'RIGHT'
        dir_u = self.direction == 'UP'
        dir_d = self.direction == 'DOWN'

        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),

            # Danger right
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),

            # Danger left
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),

            # Move direction
            dir_l, dir_r, dir_u, dir_d,

            # Food location 
            self.food_pos[0] < self.snake_pos[0],  # Food left
            self.food_pos[0] > self.snake_pos[0],  # Food right
            self.food_pos[1] < self.snake_pos[1],  # Food up
            self.food_pos[1] > self.snake_pos[1],  # Food down
        ]

        return np.array(state, dtype=int)
def train(model, replay_buffer, batch_size):
    minibatch = replay_buffer.sample(batch_size)
    if not minibatch:
        return

    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + 0.99 * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

# Main function to run the game with AI
def main():
    game = SnakeGameAI()
    model_file = 'snake_model.keras'  # 11 inputs, 4 outputs (action space)
    if os.path.exists(model_file):
        print("Loading existing model...")
        model = load_model(model_file)
    else:
        print("No model found. Creating a new one...")
        input_size = 11  # Define based on your state size
        output_size = 4  # Define based on your action space size
        model = create_model(input_size, output_size)
    replay_buffer = ReplayBuffer(max_size=1000)
    episodes = 10
    batch_size = 64
    epsilon = 1.0  # Exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.995

    
    for episode in range(episodes):
        state = game.reset()
        state = np.reshape(state, [1, 11])  # Correct shape for the model
        done = False
        total_reward = 0
        best_score = 0  

        while not done:
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, 4)  # Explore action space
            else:
                q_values = model.predict(state)  # Exploit learned values
                action = np.argmax(q_values[0])

            next_state, reward, done, _ = game.play_step(action)
            next_state = np.reshape(next_state, [1, 11])  # Ensure correct shape

            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if done:
                epsilon = max(epsilon_min, epsilon_decay * epsilon)  # Decrease epsilon

        if game.score > best_score:
            best_score = game.score
            model.save("snake_model.h5")  # Save the model as the best model
            print(f"New best score: {best_score}. Model saved.")

        if len(replay_buffer.buffer) > batch_size:
            train(model, replay_buffer, batch_size)

        print(f"Episode: {episode+1}, Score: {game.score}, Total Reward: {total_reward}")

    


if __name__ == "__main__":
    main()
