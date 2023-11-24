import numpy as np
import gym
from stable_baselines3 import PPO, A2C, DQN, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from env import WareHouseEnv
import warnings

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore', category=UserWarning, module='stable_baselines3.common.vec_env.base_vec_env')

import time
# [ExcavatorEnv 코드는 여기에 위치]
 
# 환경 생성
env = DummyVecEnv([lambda: WareHouseEnv(map_size=4, max_steps=5000, graphic=0, fps=30)])

def learning_rate_schedule(progress_remaining):
    # 이 함수는 학습의 진행에 따라 학습률을 반환합니다.
    # progress_remaining 값은 1에서 시작하여 0으로 줄어듭니다.
    return 0.001 * progress_remaining

model = PPO("MlpPolicy", env, verbose=0, learning_rate=learning_rate_schedule)
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