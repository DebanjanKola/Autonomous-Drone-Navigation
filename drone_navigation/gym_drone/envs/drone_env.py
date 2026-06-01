
"""
Custom Gym environment adapted from the car example to a drone navigating a 2D city map.
- Discrete grid state space derived from the provided city_map image (/mnt/data/city_map.png).
- Discrete action space: 8-connected moves (N, S, E, W, NE, NW, SE, SW).
- Reward: small step penalty, large positive reward for goal, negative for collision/out-of-bounds.
- Rendering overlays the drone image (/mnt/data/drone.jpg) on the map and returns an RGB frame.

Key design decisions / simplifications:
- The drone flies overhead on a 2D map; buildings/obstacles are detected from map alpha or brightness threshold.
- State is the flattened grid cell index (r * cols + c) so classic tabular Q-learning applies.
- The environment exposes `grid` and `cell_size` attributes for debugging/training.
"""

import gym
from gym import spaces
import numpy as np
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt


class DroneGridEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, map_path='assets/city_map.png', drone_path='assets/drone.jpg',
                 cell_size=16, max_steps=500, obstacle_thresh=100):
        super().__init__()
        # Load map image (RGBA or RGB). Convert to grayscale for obstacle detection.
        assert os.path.exists(map_path), f"Map image not found: {map_path}"
        self.map_img = Image.open(map_path).convert('RGBA')
        self.map_rgba = self.map_img.copy()
        self.map_gray = ImageOps.grayscale(self.map_img)

        # Load drone icon for rendering
        assert os.path.exists(drone_path), f"Drone image not found: {drone_path}"
        self.drone_img = Image.open(drone_path).convert('RGBA')

        # Discretize the map into a grid of cells (cell_size x cell_size pixels)
        self.cell_size = cell_size
        self.cols = self.map_img.width // cell_size
        self.rows = self.map_img.height // cell_size
        self.grid_shape = (self.rows, self.cols)

        # Build occupancy grid: True if obstacle (building) present in that cell
        gray = np.array(self.map_gray)
        self.obstacle_thresh = obstacle_thresh
        self.grid = np.zeros(self.grid_shape, dtype=np.uint8)
        for r in range(self.rows):
            for c in range(self.cols):
                y0 = r * cell_size
                x0 = c * cell_size
                patch = gray[y0:y0 + cell_size, x0:x0 + cell_size]
                # compute mean brightness; assume darker areas are obstacles (buildings)
                if patch.mean() < self.obstacle_thresh:
                    self.grid[r, c] = 1  # obstacle

        # Define action space: 8 moves (dx, dy)
        self._actions = [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1),   # right
            (-1, -1), # up-left
            (-1, 1),  # up-right
            (1, -1),  # down-left
            (1, 1),   # down-right
        ]
        self.action_space = spaces.Discrete(len(self._actions))

        # Observation is discrete cell index
        self.observation_space = spaces.Discrete(self.rows * self.cols)

        # Episodes
        self.max_steps = max_steps
        self.current_step = 0

        # placeholders set in reset
        self.start_pos = None
        self.goal_pos = None
        self.agent_pos = None

        # render helper
        self._last_frame = None

    def _cell_to_pixel(self, cell):
        r, c = cell
        # center pixel
        y = r * self.cell_size + self.cell_size // 2
        x = c * self.cell_size + self.cell_size // 2
        return x, y

    def _is_valid(self, cell):
        r, c = cell
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return False
        if self.grid[r, c] == 1:
            return False
        return True

    def _random_free_cell(self):
        free = np.argwhere(self.grid == 0)
        idx = np.random.choice(len(free))
        r, c = free[idx]
        return int(r), int(c)

    def reset(self, start=None, goal=None, seed=None, options=None):
        # choose random free start and goal if not specified
        if start is None:
            self.start_pos = self._random_free_cell()
        else:
            self.start_pos = tuple(start)
        if goal is None:
            # ensure goal is not equal to start
            self.goal_pos = self._random_free_cell()
            while self.goal_pos == self.start_pos:
                self.goal_pos = self._random_free_cell()
        else:
            self.goal_pos = tuple(goal)

        self.agent_pos = tuple(self.start_pos)
        self.current_step = 0

        obs = self._pos_to_state(self.agent_pos)
        return obs

    def _pos_to_state(self, pos):
        r, c = pos
        return r * self.cols + c

    def _state_to_pos(self, state):
        r = state // self.cols
        c = state % self.cols
        return int(r), int(c)

    def step(self, action):
        self.current_step += 1
        done = False
        info = {}

        # apply action
        dr, dc = self._actions[action]
        nr = self.agent_pos[0] + dr
        nc = self.agent_pos[1] + dc
        new_pos = (nr, nc)

        # check bounds and obstacles
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            # out of bounds
            reward = -10.0
            done = False
            # do not move the agent (or you could wrap it)
        elif self.grid[nr, nc] == 1:
            # collision with obstacle
            reward = -50.0
            done = True  # end episode on collision; configurable
            self.agent_pos = new_pos
        else:
            # valid move
            self.agent_pos = new_pos
            # small step penalty
            reward = -1.0

            # check goal
            if self.agent_pos == self.goal_pos:
                reward = 100.0
                done = True

        # check max steps
        if self.current_step >= self.max_steps:
            done = True

        obs = self._pos_to_state(self.agent_pos)
        return obs, float(reward), bool(done), info

    def render(self, mode='human'):
        # Overlay the drone image at the agent position on the map and return frame
        base = self.map_rgba.copy()
        frame = base.convert('RGBA')

        # compute pixel center for drone
        x, y = self._cell_to_pixel(self.agent_pos)
        # paste drone (centered). Resize drone icon to cell_size for aesthetics
        drone_icon = self.drone_img.resize((self.cell_size, self.cell_size), Image.LANCZOS)
        w, h = drone_icon.size
        # top-left
        tlx = int(x - w / 2)
        tly = int(y - h / 2)

        frame.paste(drone_icon, (tlx, tly), drone_icon)

        # optionally draw start and goal markers
        start_x, start_y = self._cell_to_pixel(self.start_pos)
        goal_x, goal_y = self._cell_to_pixel(self.goal_pos)
        # draw small colored circles using matplotlib overlay
        self._last_frame = frame.convert('RGB')

        if mode == 'rgb_array':
            return np.array(self._last_frame)
        elif mode == 'human':
            plt.imshow(self._last_frame)
            plt.axis('off')
            plt.pause(0.001)
            return None

    def close(self):
        plt.close()
