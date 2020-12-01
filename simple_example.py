import gym
import gym_snake

# Construct Environment
env = gym.make('snake-v0')
env.grid_size = [10, 10]
env.snake_size = 3
env.unit_size = 1
env.unit_gap = 0

observation = env.reset() # Constructs an instance of the game

for _ in range(1000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
    # BODY_COLOR = np.array([1, 0, 0], dtype=np.uint8)
    # HEAD_COLOR = np.array([255, 10 * i, 0], dtype=np.uint8)
    # SPACE_COLOR = np.array([0, 255, 0], dtype=np.uint8)
    # FOOD_COLOR = np.array([0, 0, 255], dtype=np.uint8)
    if done:
        env.reset()
env.close()


