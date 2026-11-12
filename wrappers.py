import gym
import numpy as np
import cv2


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.shape[0], self.shape[1], 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)


class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_score = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # El hack para ganar: recompensa adicional si el score sube
        score_diff = info['score'] - self.current_score
        if score_diff > 0:
            reward += (score_diff / 10.0)

        self.current_score = info['score']

        if done:
            self.current_score = 0

        return obs, reward, done, info

    def reset(self):
        self.current_score = 0
        return self.env.reset()