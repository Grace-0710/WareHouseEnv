import numpy as np
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from env2 import WareHouseEnv
from dqn import DQN  # Assuming your DQN class is defined in a file named dqn.py
import warnings

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore', category=UserWarning, module='stable_baselines3.common.vec_env.base_vec_env')

import time
# 환경 생성
env = DummyVecEnv([lambda: WareHouseEnv(map_size=4, max_steps=5000, graphic=0, fps=30)])

# Learning rate schedule function
def learning_rate_schedule(progress_remaining):
    return 0.001 * progress_remaining

model = DQN("MlpPolicy", env, verbose=0, learning_rate=learning_rate_schedule)
model.learn(total_timesteps=100_000_000)

# 학습된 모델을 환경에서 테스트
obs = env.reset()
for _ in range(100_000_000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()
    # env.envs[0].render()
    # time.sleep(0.3)



env.close()
