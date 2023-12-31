# WareHouseEnv
강화학습 프로젝트 
- 팀원 : 신요섭, 정승연, 윤서진
- 우리의 목표 : 2개의 지게차와 3개의 로봇으로 150개의 짐을 최적의 경로로 옮겨주세요
1. 현재 env.py를 DQN, Monte-Carlo, TD learning 등 수업에서 배운 코드를 통해 학습을 할 수 있도록 한다.
2. 1단계가 완수가 되면, 학습이 잘 되도록 하기 위해서 reward function을 변경하며 적용한다.
3. 각 단계를 진행하며 정리된 내용을 토대로 보고서를 작성한다.

2개의 지게차와 3개의 로봇이 있습니다
지게차와 로봇들의 첫 위치는 지정이 되어 있으며 랜덤으로도 바꿀 수 있습니다
* 클래스 정의:
    * WareHouseEnv 클래스는 OpenAI Gym의 gym.Env를 상속합니다.
    * "지게차" 시뮬레이션을 모델링하며, 코드에서는 로봇과 리프트가 작업을 수행하는 환경을 구현합니다.
* 환경 구성:
    * self.action_space: 로봇과 리프트의 행동 공간을 정의합니다. 여러 로봇이 각각 4개의 가능한 행동을 취할 수 있습니다.
    * self.observation_space: "지게차"의 상태를 나타내는 2차원 배열로 정의됩니다.
    * self.cargo_map: "지게차"의 초기 상태를 설정하며, 로봇과 리프트가 작업할 지역을 모델링합니다.
    * 리프트와 로봇의 초기 위치를 설정하고, 로봇이나 리프트가 적재한 물체의 개수를 나타내는 변수를 초기화합니다.
* 주요 메서드:
    * reset(): 환경을 초기 상태로 리셋하고, 초기 관측값을 반환합니다.
    * _next_observation(): 현재 "지게차"의 상태를 반환합니다.
    * step(actions): 주어진 행동에 대한 다음 상태로 진행하고, 보상과 종료 여부를 반환합니다.
    * render_frame(): "지게차" 상태를 시각적으로 표시합니다.
    * close(): 시각화 창을 닫습니다.
* 시각화:
    * Pygame 라이브러리를 사용하여 "지게차" 상태를 시각적으로 표시합니다. render_frame 메서드에서 시각화가 이루어집니다.
* 초기화 및 활용:
    * 환경은 초기화 후 Gym 환경처럼 사용할 수 있습니다. reset()으로 초기화하고, step()으로 행동을 수행하며 환경을 진행합니다.
      
[ 프로젝트 조건 ]
- 리워드 증가 : 로봇이 짐을 많은 곳으로 이동하는 경우 
  * 조건: 로봇이 이동한 위치에 짐이 있는 경우
  * 리워드: 로봇이 짐을 이동한 위치의 짐을 버리면 리워드가 증가
    
- 패널티 부여
1. 로봇이 지게차의 짐을 버리지 않거나 이동하지 않는 경우 
   * 조건: 로봇이 이동한 위치에 짐이 없거나 로봇이 이동하지 않은 경우
   * 패널티: 짐을 버리지 않거나 이동하지 않은 경우에 패널티 부여
2. 로봇이 이동했는데 짐이 없는 곳으로 이동하는 경우
   * 조건: 로봇이 이동한 위치에 짐이 없는 경우
   * 패널티: 로봇이 이동한 위치에 짐이 없는 경우에 패널티 부여
 3. 짐의갯수가 적을수록 패널티 부여정도가 낮다 ( 짐의 갯수에 비해 반비례 )
    
- 목표 : 로봇의 최소 움직임으로 모든 짐이 없어져야 함.

[ 프로젝트 파일 ]
- 환경 : anaconda로 가상환경 구축 후 실행
- WareHouseEnv 환경구축 파일 : env.py
- dqn 파일 : dqn_main.py
- custom dqn 파일 : ex16_dqn_main.py
- Train Model Download File
   * custom dqn 파일 : [EX16_DQN_MODEL.zip](https://github.com/Grace-0710/WareHouseEnv/blob/master/EX16_DQN_MODEL.zip)
   * dqn 파일 : [DQN_Model.zip](https://github.com/Grace-0710/WareHouseEnv/blob/master/DQN_Model.zip)
   * A2C 파일 : [A2C_Model.zip](https://github.com/Grace-0710/WareHouseEnv/blob/master/A2C_Model.zip)
   * PPO 파일 : [PPO_Model.zip](https://github.com/Grace-0710/WareHouseEnv/blob/master/PPO_Model.zip)
[ 프로젝트 빌드 ]
1. requirements.txt install : pip install -r requirements.txt ( pip3 install -r requirements.txt )
2. train 학습 run
- dqn 알고리즘 학습 : python dqn_main.py ( python3 dqn_main.py )
- Custom dqn 알고리즘 학습 : python ex16_dqn_main.py ( python3 ex16_dqn_main.py )
- A2C 알고리즘 학습 : python a2c_main.py ( python3 a2c_main.py )
- PPO 알고리즘 학습 : python ppo_main.py ( python3 ppo_main.py )
