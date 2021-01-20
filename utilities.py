
import numpy as np

# obs space [map_width, map_height, channel_depth], channel values are:
    #   BODY_COLOR = np.array([1, 0, 0], dtype=np.uint8)
    #   HEAD_COLOR = np.array([255, 10 * i, 0], dtype=np.uint8)
    #   FOOD_COLOR = np.array([0, 0, 255], dtype=np.uint8)
    #   SPACE_COLOR = np.array([0, 255, 0], dtype=np.uint8)

# transform into
# height x width x [head, body, food]
    # empty space = [0, 0, 0]
    # head = [1, 0, 0]
    # body = [0, 1, 0]
    # food = [0, 0, 1]

colors = {
        tuple([255, 10, 0]) : np.array([1, 0, 0]), # HEAD_COLOR to head
        tuple([1, 0, 0]) : np.array([0, 1, 0]), # BODY_COLOR to body
        tuple([0, 0, 255]) : np.array([0, 0, 1]), # FOOD_COLOR to food
        tuple([0, 255, 0]) : np.array([0, 0, 0]), # SPACE_COLOR to space
    }

def normalize(obs):
    obs = obs.astype(np.float32)
    nobs = np.zeros_like(obs)
    for x in range(obs.shape[0]):
        for y in range(obs.shape[1]):
            nobs[x,y] = colors[tuple(obs[x,y])]
    return nobs

def standarize(obs):
    obs = (obs - obs.mean()) / obs.var()
    return obs

# make obs zero-centered, with std = 1
def normalize_and_flatten(obs):
    return normalize(obs).flatten()


