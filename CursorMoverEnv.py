import sys
from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
from pygame import gfxdraw
import random


class CursorMoverEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, config, render_mode: Optional[str] = None):
        self.grid_size = config["grid_size"]
        self.cardinality = config["cardinality"]
        self.stage_size = config["stage_size"]
        self.render_wait = config["render_wait"]
        self.episode_length = config["episode_length"]
        self.center_dot = config["center_dot"]
        self.image_path = config["image_path"]
        self.images = {}
        self.images[0] = pygame.image.load(self.image_path + "club.jpg")
        self.images[1] = pygame.image.load(self.image_path + "heart.jpg")
        self.images[2] = pygame.image.load(self.image_path + "diamond.jpg")
        self.images[3] = pygame.image.load(self.image_path + "spade.jpg")
        self.suits = [0, 1, 2, 3]
        self.cursors = {}
        self.cursors[0] = pygame.image.load(self.image_path + "pink_cursor.png")
        self.cursors[1] = pygame.image.load(self.image_path + "blue_cursor.png")

        self.scene_size = self.stage_size * 2 - 1
        self.scene_image_size = self.scene_size * self.grid_size
        self.max_jump = self.scene_size - self.stage_size
        self.pos_xy = np.zeros((self.cardinality, 2), dtype=np.int8)
        self.cursor_xy = np.zeros((2, 2), dtype=np.int8)
        self.object_size = np.ones(self.cardinality, dtype=np.int8)
        self.brightness = np.ones(self.cardinality, dtype=np.int8)

        self.action_space = spaces.Box(low=-1 * self.max_jump, high=self.max_jump, shape=(5,), dtype=np.int8)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.scene_image_size, self.scene_image_size, 3),
                                            dtype=np.uint8)
        self.observation = np.zeros((self.scene_image_size, self.scene_image_size, 3), dtype=np.uint8)
        self.gaze = np.array([0, 0])
        self.render_mode = render_mode
        self.isopen = True
        self.episode_count = 0
        self.dump = config['dump']

        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.scene_image_size, self.scene_image_size))
        self.scene = pygame.Surface((self.scene_image_size, self.scene_image_size))
        self.clock = pygame.time.Clock()

    def step(self, action):
        saccade = np.array(action[:2])
        cursor_action = np.array(action[2:4])
        cursor_shift = [0, 0]
        if 0 <= self.cursor_xy[1][0] + cursor_action[0] < self.stage_size:
            cursor_shift[0] = cursor_action[0]
        if 0 <= self.cursor_xy[1][1] + cursor_action[1] < self.stage_size:
            cursor_shift[1] = cursor_action[1]
        grab = action[4]
        drag = 0    # for logging
        # print("saccade:", saccade, " gaze:", self.gaze)
        self.gaze = self.gaze + saccade
        self.screen = pygame.display.set_mode((self.scene_image_size, self.scene_image_size))
        self.scene = pygame.Surface((self.scene_image_size, self.scene_image_size))
        stage_corner = (self.scene_size - self.stage_size) // 2
        for i in range(self.cardinality):
            if grab == 1 and np.all(self.pos_xy[i] == self.cursor_xy[1]):
                self.pos_xy[i] = self.cursor_xy[1] + cursor_shift
                drag = 1    # for logging
            left = ((self.scene_size - self.stage_size) // 2 + self.pos_xy[i, 0]) * self.grid_size
            top = ((self.scene_size - self.stage_size) // 2 + self.pos_xy[i, 1]) * self.grid_size
            self.scene.blit(self.images[self.suits[i]], (left, top))
        self.cursor_xy[1] = self.cursor_xy[1] + cursor_shift
        for i in range(2):
            left = ((self.scene_size - self.stage_size) // 2 + self.cursor_xy[i, 0] + 1) * self.grid_size - 13
            top = ((self.scene_size - self.stage_size) // 2 + self.cursor_xy[i, 1] + 1) * self.grid_size - 16
            self.scene.blit(self.cursors[i], (left, top))
        pygame.draw.rect(self.scene, (100,100,100), pygame.Rect(stage_corner * self.grid_size,
                                                                stage_corner * self.grid_size,
                                                                self.stage_size * self.grid_size,
                                                                self.stage_size * self.grid_size), 1)
        img = pygame.surfarray.array3d(self.scene)
        gaze_offset = self.gaze * self.grid_size
        self.observation = np.zeros((self.scene_image_size, self.scene_image_size, 3), dtype=np.uint8)
        if self.gaze[0] < 0:
            if self.gaze[1] < 0:
                self.observation[-1 * gaze_offset[0]:,
                                 -1 * gaze_offset[1]:] =\
                    img[:self.scene_image_size + gaze_offset[0],
                        :self.scene_image_size + gaze_offset[1]]
            else:
                self.observation[-1 * gaze_offset[0]:,
                                 0:self.scene_image_size - gaze_offset[1]] =\
                    img[:self.scene_image_size + gaze_offset[0], gaze_offset[1]:]
        else:
            if self.gaze[1] < 0:
                self.observation[0:self.scene_image_size - gaze_offset[0],
                                 -1 * gaze_offset[1]:] =\
                    img[gaze_offset[0]:, :self.scene_image_size + gaze_offset[1]]
            else:
                self.observation[0:self.scene_image_size - gaze_offset[0],
                                 0:self.scene_image_size - gaze_offset[1]] = \
                    img[gaze_offset[0]:, gaze_offset[1]:]
        self.episode_count += 1
        if self.dump is not None:
            self.dump.write("Env: #{0}, action {1}, {2}\n"
                            .format(self.episode_count, action, drag))
        if self.episode_count >= self.episode_length:
            return self.observation, 0, True, False, {}
        else:
            return self.observation, 0, False, False, {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        obs_surf = pygame.surfarray.make_surface(self.observation)
        self.screen.blit(obs_surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()
        pygame.time.wait(self.render_wait)

    def reset(self, seed=None, options={}):
        self.gaze = np.array([0, 0])
        self.episode_count = 0
        pygame.init()
        self.scene.fill((0, 0, 0))
        if self.center_dot:
            gfxdraw.filled_circle(self.scene, self.scene_image_size // 2, self.scene_image_size // 2, 3, (255, 0, 0))
        self.observation = np.zeros((self.scene_image_size, self.scene_image_size, 3), dtype=np.uint8)
        random.shuffle(self.suits)
        positions = np.random.choice(self.stage_size * self.stage_size, size=self.cardinality, replace=False)
        for i in range(self.cardinality):
            self.pos_xy[i, 0] = positions[i] % self.stage_size
            self.pos_xy[i, 1] = positions[i] // self.stage_size
        cursor_positions = np.random.choice(self.stage_size * self.stage_size, 2, replace=False)
        cursor_positions[1] = positions[np.random.choice(self.cardinality, 1)[0]]  # initially on one of the suites
        for i in range(2):
            self.cursor_xy[i, 0] = cursor_positions[i] % self.stage_size
            self.cursor_xy[i, 1] = cursor_positions[i] // self.stage_size
        return self.observation, {}

    def close(self):
        pygame.display.quit()
        pygame.quit()
        self.isopen = False


def run(env, gaze):
    print(gaze)
    env.reset()
    action = np.concatenate([gaze, [1, 1, 0]])
    env.step(action)
    env.render()


def main():
    config = {
        "stage_size": 3,
        "grid_size": 28,
        "cardinality": 1,
        "render_wait": 1000,
        "episode_length": 3,
        "center_dot": True,
        "image_path": "env/28×28/"
    }
    if config["cardinality"] > config["stage_size"] * config["stage_size"]:
        print('Error: cardinality cannot be larger than stage_size^2!', file=sys.stderr)
        sys.exit(1)
    env = CursorMoverEnv(config, render_mode="human")
    # gaze = np.random.randint(-2, 3, size=2)
    print(isinstance(env.action_space, spaces.Box))
    run(env, np.array([0, 0]))
    run(env, np.array([0, 0]))
    run(env, np.array([0, 0]))
    run(env, np.array([0, 0]))
    run(env, np.array([0, 0]))
    # run(env, np.array([1, 1]))
    # run(env, np.array([-1, -1]))
    # run(env, np.array([-1, 1]))
    # run(env, np.array([1, -1]))
    env.close()


if __name__ == '__main__':
    main()
