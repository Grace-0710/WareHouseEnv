import numpy as np
import gym
from stable_baselines3 import PPO, A2C, DQN, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from env2 import WareHouseEnv
import warnings
import matplotlib.pyplot as plt
# 경고 메시지 무시 설정
warnings.filterwarnings('ignore', category=UserWarning, module='stable_baselines3.common.vec_env.base_vec_env')

import time
# [ExcavatorEnv 코드는 여기에 위치]
 
# 환경 생성
env = DummyVecEnv([lambda: WareHouseEnv(map_size=4, max_steps=5000, graphic=0, fps=30)])

# A2C 모델 초기화
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000_000)

# 학습된 모델을 환경에서 테스트
obs = env.reset()
reward_return_list = []
for _ in range(100_000_000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    reward_return_list.append(rewards)



env.close()
plt.plot(reward_return_list)
plt.xlabel('Iteration')
plt.ylabel('Reward A2C')
plt.savefig('Reward_A2C.png', format='png', dpi=300)
# Display the plot
plt.show()