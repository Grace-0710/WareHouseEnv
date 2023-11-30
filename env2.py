import numpy as np
import gym
from gym import spaces
import pygame

class WareHouseEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_size=5, max_steps=2000, graphic=True, fps=150):
        super(WareHouseEnv, self).__init__()

        self.episode = 0
        self.map_size = map_size
        self.log_map = []

        self.max_steps = max_steps
        self.current_step = 0
        
        self.num_lift = 2
        self.num_robot = 3
        num_actions_per_agent = 5  # 'stay' 액션을 포함한 액션 수
        total_agents = self.num_lift + self.num_robot
        total_combinations = num_actions_per_agent ** total_agents

        self.action_space = spaces.Discrete(total_combinations)
        self.observation_space = spaces.Box(low=0, high=5, shape=(map_size, map_size), dtype=np.int32)

        self.desired_total_sum = 150
        self.cargo_map = self.generate_fixed_sum_array((map_size, map_size), self.desired_total_sum)
        self.cargo_map[0][0] = 0  # Start position for the soil bank

        self.lift_positions = [[2, 2], [2, 3]]
        self.robot_positions = [[3, 3], [3, 2], [3, 1]]
        
        self.lift_carry = [0] * self.num_lift
        self.robot_load = [0] * self.num_robot
        self.reward = 0

        self.render_mode = 'human'
        self.window = None
        self.clock = None
        self.draw_fps = fps
        self.cell_size = 90
        self.win_size = self.cell_size * self.map_size
        self.graphic = graphic

    def reset(self):
        self.current_step = 0
        self.cargo_map = self.generate_fixed_sum_array((self.map_size, self.map_size), self.desired_total_sum)
        self.cargo_map[0][0] = 0
        self.lift_carry = [0] * self.num_lift
        self.robot_load = [0] * self.num_robot
        self.reward = 0
        return self._next_observation()

    def _next_observation(self):
        return self.cargo_map

    def step(self, action):
        # Decode the single integer action into individual actions for each agent
        actions = self._decode_action(action)

        reward, done = self._apply_actions(actions)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            self.episode += 1

        if self.graphic and self.render_mode == 'human':
            self.render_frame()

        return self._next_observation(), self.reward, done, {}

    def _decode_action(self, action):
        """
        Decode the single integer action into a list of individual actions for each agent.
        """
        actions = []
        for _ in range(self.num_lift + self.num_robot):
            actions.append(action % 5)
            action //= 5
        return actions[::-1]

    def _apply_actions(self, actions):
        reward = 0
        done = False
        lift_penalty = 0
        robot_penalty = 0
        lift_bonus = 0

        # [여기에 각 에이전트에 대한 액션 적용 로직을 작성하세요]
        # [리프트와 로봇의 이동, 상호작용, 보상 계산 등]

        # Check if all the cargo has been moved
        if np.sum(self.cargo_map) == 0:
            reward += 2000 - self.current_step
            done = True
            self.episode += 1

        elif self.current_step >= self.max_steps:
            reward -= self.current_step
            done = True
            self.episode += 1

        return reward, done


    def render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.win_size, self.win_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.win_size, self.win_size))
        canvas.fill((255, 255, 255))

        # 수화물 크기
        cargo_pix_size = 9
        # workspace의 구분선 크기
        line_width = 3

        for y in range(self.map_size):
            for x in range(self.map_size):
                n_soil = self.cargo_map[y][x]
                for z in range(n_soil):
                    # Calculate the top-left corner of the soil cell
                    top_left_x = x * self.cell_size + z * cargo_pix_size + line_width
                    top_left_y = y * self.cell_size + line_width
                    # Draw the soil cell
                    pygame.draw.rect(canvas, (146, 104, 41), pygame.Rect(top_left_x, top_left_y, cargo_pix_size, cargo_pix_size))
                    # Draw the grid lines around the soil cell
                    pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(top_left_x - line_width, top_left_y - line_width, cargo_pix_size + line_width, cargo_pix_size + line_width), width=line_width)


        pygame.font.init()
        font = pygame.font.SysFont(None, 25)  # 기본 폰트와 크기 25로 설정

        lift_pix_size = 30
        for i in range(self.num_lift):
            rows, cols = self.lift_positions[i][1], self.lift_positions[i][0]
            color = (0, 255, 0) if i == 1 else (0, 0, 255)
            carry_value = self.lift_carry[i]
            lift_y_offset = self.cell_size / 3
            lift_x_offset = (i % 3) * (self.cell_size / 3)
            carry_text = font.render(str(carry_value), True, (255, 255, 255))
            text_x = cols*self.cell_size + lift_x_offset + lift_pix_size/2 - carry_text.get_width() / 2
            text_y = rows*self.cell_size + lift_y_offset + lift_pix_size/2 - carry_text.get_height() / 2
            excavator_rect = pygame.Rect(
                cols*self.cell_size + lift_x_offset,
                rows*self.cell_size + lift_y_offset,
                lift_pix_size,
                lift_pix_size
            )
            pygame.draw.rect(canvas, color, excavator_rect)
            canvas.blit(carry_text, (text_x, text_y))


        
        lift_pix_size = 30
        for i in range(self.num_robot):
            rows, cols = self.robot_positions[i][1], self.robot_positions[i][0]
            color = (0, 0, 0)
            
            robot_y_offset = (self.cell_size / 3) * 2
            pygame.draw.rect(canvas, color, pygame.Rect(cols*self.cell_size + self.cell_size/2 - lift_pix_size/2, rows*self.cell_size + robot_y_offset, lift_pix_size, lift_pix_size))
            
            load_text = font.render(str(self.robot_load[i]), True, (255, 255, 255))
            canvas.blit(load_text, (cols*self.cell_size + self.cell_size/2 - load_text.get_width()/2, rows*self.cell_size + robot_y_offset + lift_pix_size/2 - load_text.get_height()/2))

        for x in range(self.map_size + 1):
            pygame.draw.line(canvas, (0, 0, 0), (0, self.cell_size*x), (self.win_size, self.cell_size*x), width=line_width)
            pygame.draw.line(canvas, (0, 0, 0), (self.cell_size*x, 0), (self.cell_size*x, self.win_size), width=line_width)

        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.draw_fps)
            
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def generate_fixed_sum_array(self, shape, total_sum):
        arr = np.random.rand(*shape)
        ratio = arr / arr.sum()
        arr = (ratio * total_sum).astype(int)
        diff = total_sum - arr.sum()
        while diff != 0:
            for i in range(abs(diff)):
                idx = np.random.randint(0, np.prod(shape))
                i, j = np.unravel_index(idx, shape)
                if diff > 0 and arr[i, j] < 10:
                    arr[i, j] += 1
                elif diff < 0 and arr[i, j] > 0:
                    arr[i, j] -= 1
            diff = total_sum - arr.sum()
        return arr
