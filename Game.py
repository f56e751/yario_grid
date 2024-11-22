import pygame
import retro
import numpy as np
from utils import SMB
from PIL import Image
from Visualizer import GameFrameVisualizer
from tensor import Tensor
from yolo_class_mapping import ClassMapping
import torch
import time
import sys
import os

import torch
from torchvision import transforms


class Game():
    def __init__(self, x_pixel_num, y_pixel_num, visualize, window_x = 500, window_y = 500, center = True):
        # self.env = retro.make(game='SuperMarioBros-Nes', state = 'Level3-1')
        self.env = retro.make(game='SuperMarioBros-Nes', state = 'Level1-1')
        self.env.reset()
        # Pygame 설정
        pygame.init()

        # if visualize:
        #     if not center:
        #         os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"
        #     else:
        #         screen_info = pygame.display.Info()
        #         screen_width, screen_height = screen_info.current_w, screen_info.current_h
                
        #         # 창의 중앙 위치 계산
        #         window_x = (screen_width - x_pixel_num) // 2
        #         window_y = (screen_height - y_pixel_num) // 2
        #         os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"
            
        
        

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.x_pixel_num = x_pixel_num
        self.y_pixel_num = y_pixel_num
        self.visualize = visualize
        if self.visualize:
            self.game_screen = pygame.display.set_mode((self.x_pixel_num, self.y_pixel_num))
        # self.game_screen = pygame.display.set_mode((512, 480))
        self.clock = pygame.time.Clock()

        self.is_new_action_received = False
        self.new_action = None
        self.ram = None
        self.fps = 60
        self.null_action = np.array( [0, 1,    0,      0,     0, 0, 0, 0, 0], np.int8)

        self.gameFrameVisualizer = GameFrameVisualizer()
        if self.visualize:
            self.gameFrameVisualizer.set_game_screen()
        self.running = True

        self.frame_count = 0 # 이 값을 기준으로 텐서 반환 시점을 정함
        self.tile_info = {}
        self.elapsed_frame_return_tile_info = 0 # get_tile_info가 호출될 때 이 값에 self.elapsed_frame를 대입함
        self.elapsed_frame_num = 0 # update_game이 호출될 때마다 1 증가함
        self.tensor = Tensor()
        self.class_mapping = ClassMapping()

        self.previous_action = np.array( [0, 1,    0,      0,     0, 0, 0, 0, 0], np.int8)
        self.prev_mario_state = 0
        self.prev_score = 0
        self.prev_mario_x = 0
        self.action_map = { 0: np.array( [0, 1,    0,      0,     0, 0, 0, 0, 0], np.int8),
                    1: np.array( [0, 0,    0,      0,     0, 1, 0, 0, 0], np.int8),
                    2: np.array( [0, 0,    0,      0,     0, 0, 1, 0, 0], np.int8),
                    3: np.array( [0, 0,    0,      0,     0, 0, 0, 1, 0], np.int8),
                    4: np.array( [0, 0,    0,      0,     0, 0, 0, 0, 1], np.int8),
                    5: np.array( [1, 0,    0,      0,     0, 0, 0, 0, 0], np.int8),
                    6: np.array( [0, 0,    0,      0,     0, 0, 0, 1, 1], np.int8),
                    7: np.array( [0, 0,    0,      0,     0, 0, 1, 0, 1], np.int8),
                    8: np.array( [1, 0,    0,      0,     0, 0, 0, 1, 0], np.int8),
                    9: np.array( [1, 0,    0,      0,     0, 0, 1, 0, 0], np.int8),
                    10: np.array( [1, 0,    0,      0,     0, 0, 0, 1, 1], np.int8),
                    11: np.array( [1, 0,    0,      0,     0, 0, 1, 0, 1], np.int8),
                    }

        # self.transform = transforms.ToTensor()

        self.transform = transforms.Compose([
                    transforms.Grayscale(),  # 그레이스케일 변환
                    transforms.ToTensor()    # 텐서 변환
                ])

    def stop(self):
        self.running = False

    def update_game(self, action):
        # action이 np.array([0] * 9) 형태가 아닌 경우 에러 발생
        if not isinstance(action, np.ndarray) or action.shape != (9,):
            raise ValueError("The action must be a numpy array of shape (9,).")

        # # Keys correspond with             B, NULL, SELECT, START, U, D, L, R, A
        # # index                            0  1     2       3      4  5  6  7  8
        # self.buttons_to_press = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)


        # 게임 환경 업데이트
        obs, rew, done, info = self.env.step(action)
        self.elapsed_frame_num += 1
        if done:
            self.env.reset()

        # self.visualize_frame()

    def visualize_frame(self):
        if self.visualize:
            pygame.event.pump()
                

            rendered_frame = self.env.render(mode='rgb_array').swapaxes(0, 1)
            frame = pygame.surfarray.make_surface(rendered_frame)
            frame_scaled = pygame.transform.scale(frame, (self.x_pixel_num, self.y_pixel_num))

            self.game_screen.blit(frame_scaled, (0, 0))
            pygame.display.flip()
            # self.game_screen.blit(pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1)), (0, 0))
            # # 게임 화면 스케일링
            # frame = pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1))
            # frame = pygame.transform.scale(frame, (self.x_pixel_num, self.y_pixel_num))  # 스케일링
            # self.game_screen.blit(frame, (0, 0))
            # pygame.display.flip()


    def update_game_human_mode(self, action):
        if not isinstance(action, np.ndarray) or action.shape != (9,):
            raise ValueError("The action must be a numpy array of shape (9,).")
        # 게임 환경 업데이트
        obs, rew, done, info = self.env.step(action)
        # print(f"obs: {obs}")
        # print(f"rew: {rew}")
        # print(f"done: {done}")
        self.elapsed_frame_num += 1
        if done:
            self.env.reset()

        self.visualize_frame()


    # train할 때 게임 환경을 업데이트하는 함수
    def step(self, action):
        if not isinstance(action, np.ndarray) or action.shape != (9,):
            raise ValueError("The action must be a numpy array of shape (9,).")
        # 게임 환경 업데이트
        obs, rew, done, info = self.env.step(action)
        # rew 값이 계속 증가하지 않고 0, 1, 2 값만 반환함 -> 
        # reward = 

        reward = self.get_reward()
        is_world_cleared = self.is_world_cleared()
        is_dead = self.is_dead()

        # 월드를 클리어했거나 죽었으면 시작지점으로 게임을 초기화
        if is_world_cleared or is_dead:
            self.reset()

        # # 게임 화면 업데이트
        # self.game_screen.blit(pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1)), (0, 0))

        # # 게임 화면 스케일링
        # frame = pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1))
        # frame = pygame.transform.scale(frame, (self.x_pixel_num, self.y_pixel_num))  # 스케일링
        # self.game_screen.blit(frame, (0, 0))
        # pygame.display.flip()
        self.visualize_frame()



        return reward, is_world_cleared, None
        ###########################################

    def step_new_ppo(self,action):
        # action은 정수
        action_np = self.action_map[action]
        obs, rew, done, info = self.env.step(action_np)
        # rew 값이 계속 증가하지 않고 0, 1, 2 값만 반환함 -> 
        # reward = 

        reward = self.get_reward()
        is_world_cleared = self.is_world_cleared()
        is_dead = self.is_dead()

        # 월드를 클리어했거나 죽었으면 시작지점으로 게임을 초기화
        if is_world_cleared or is_dead:
            self.reset()

        self.visualize_frame()

        state = self.get_img_tensor()
        return state, reward, is_world_cleared, None

    def get_reward(self):
        ram = self.env.get_ram()
        reward = 0
        if self.is_dead():
            reward -= 15

        # if self.is_get_item():
        #     reward += 1000

        current_score = self.get_mario_score()   
        score_diff = current_score - self.prev_score
        self.prev_score = current_score

        reward += score_diff / 100

        # 스크린이 시작하는 지점의 값
        # 끝났을때가 3040
        mario_position = SMB.get_mario_location_in_level(ram)
        position_diff = mario_position.x - self.prev_mario_x
        # print(f"position_diff: {position_diff}")
        reward += (position_diff)

        if position_diff <= 0:
            reward -= 1
        self.prev_mario_x = mario_position.x
        # print(f"reward: {reward}")
        return reward
    
    # 마리오가 아이템을 먹은 시점에 true 반환
    def is_get_item(self):
        mario_state = self.get_mario_state()
        if mario_state > self.prev_mario_state:
            self.prev_mario_state = mario_state
            return True
        else:
            self.prev_mario_state = mario_state
            return False

    def run(self):
        # current_time = pygame.time.get_ticks()
        self.running = True 
        while self.running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # print("game class quit")
                    self.running = False
            
            if self.is_new_action_received:
                self.update_game(self.new_action)
                self.is_new_action_received = False
                self.previous_action = self.new_action
            else:
                # self.update_game(self.null_action)
                self.update_game(self.previous_action)
            
            # last_update_time = current_time

            if self.visualize:
                self.gameFrameVisualizer.visualize(self.get_frame())
            self.clock.tick(self.fps)

    def receive_action(self, action):
        # print("new action!!!")
        self.is_new_action_received = True
        self.new_action = action

        
    # # 게임을 시작으로 되돌리는 함수
    def reset(self):
        self.env.reset()

        self.elapsed_frame_num = 0
        self.frame_count = 0
        self.tile_info = {}
        self.elapsed_frame_return_tile_info = 0

        self.is_new_action_received = False
        self.new_action = None

        self.prev_score = 0
        self.prev_mario_state = 0
        self.prev_mario_x = 0

        state = self.get_img_tensor()
        return state


    def get_mario_score(self):
        ram = self.env.get_ram()
        return SMB.get_mario_score(ram)


    # # 마리오가 죽었는지 여부를 반환하는 함수
    def is_dead(self):  
        ram = self.env.get_ram()
        return SMB.is_dead(ram)
    
    # # 1-1 월드가 클리어됐는지 여부를 반환하는 함수
    def is_world_cleared(self):
        ram = self.env.get_ram()
        # print
        return SMB.is_world_cleared(ram)

    # agent의 action이 입력 가능한지 여부
    def is_playable(self):
        ram = self.env.get_ram()
        return SMB.is_recordable(ram)
    
    def get_mario_state(self):
        # 0: small, 1: big,  => 2 : firey
        ram = self.env.get_ram()
        return SMB.get_mario_state(ram)

    # visualize할떄 필요한 frame 반환
    def get_frame(self) -> pygame.surface: 
        frame = pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1))
        frame = pygame.transform.scale(frame, (self.x_pixel_num, self.y_pixel_num))  # 스케일링
        return frame
    
    def get_yolo_input_img(self) -> Image:
        frame = pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1))
        frame = pygame.transform.scale(frame, (self.x_pixel_num, self.y_pixel_num))  # 스케일링
        pil_image = Image.frombytes('RGB', (self.x_pixel_num, self.y_pixel_num), pygame.image.tostring(frame, 'RGB'))
        return pil_image
    

    # ram 정보를 이용해 grid로 출력하기 위해 타일 정보를 반환하는 함수
    def get_tile_info(self):
        
        # # 중복되는 연산을 줄이고자 같은 프레임에 이 함수가 두 번 호출되면 이전에 저장한 값을 반환함
        # if self.elapsed_frame_return_tile_info ==  self.elapsed_frame_num:
        #     return self.tile_info


        self.elapsed_frame_return_tile_info = self.elapsed_frame_num
        ram = self.env.get_ram()
        # yolo_format = SMB.get_yolo_format_new(ram)
        yolo_format = SMB.get_yolo_format_for_game(ram)
        self.tile_info = {}

        base_x_unit_length = 16 / 256
        base_y_unit_length = 16 / 240

        grid_w = 16
        grid_h = 15

        for label_value, coordinates in yolo_format.items():
            for coordinate in coordinates:
                x_yolo, y_yolo = coordinate[0] # 0~1 사이의 값
                x_unit_length, y_unit_length = coordinate[1]

                x_size = round(x_unit_length / base_x_unit_length)
                y_size = round(y_unit_length / base_y_unit_length)


                grid_x = min(max(int(x_yolo / base_x_unit_length), 0), grid_w - 1)
                grid_y = min(max(int(y_yolo / base_y_unit_length), 0), grid_h - 1)
                
                loc = (grid_x, grid_y)
                self.tile_info[loc] = label_value

                if grid_y == 2:
                    new_loc = (grid_x, grid_y-1)
                    self.tile_info[new_loc] = label_value

        return self.tile_info
    
    def get_img_tensor(self):
        frame = pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1))

        # 이미지 크기를 원래의 절반으로 스케일링
        half_x_pixel_num = self.x_pixel_num // 2
        half_y_pixel_num = self.y_pixel_num // 2
        frame = pygame.transform.scale(frame, (half_x_pixel_num, half_y_pixel_num))  # 스케일링
        pil_image = Image.frombytes('RGB', (half_x_pixel_num, half_y_pixel_num), pygame.image.tostring(frame, 'RGB'))
        
        tensor_image = self.transform(pil_image)

        tensor_image = tensor_image.unsqueeze(0)  # 배치 차원 추가
        return tensor_image

    # network에 입력할 tensor를 반환하는 함수
    # base_frame_count만큼 프레임이 지나갔을때 반환함
    def get_tensor(self):
        mario_state = self.get_mario_state()
        tile_info = self.get_tile_info()
        grid_w = 16
        grid_h = 15
        for key, value in tile_info.items():
            grid_x, grid_y = key
            class_id = value

            grid_x = int(grid_x * grid_w)
            grid_y = int(grid_y * grid_h)

            grid_x = min(max(grid_x, 0), grid_w - 1)
            grid_y = min(max(grid_y, 0), grid_h - 1)

            group_id = self.class_mapping.get_group_id(class_id)
            # print(group_id)
            self.tensor.update(mario_state, grid_x, grid_y, group_id, self.frame_count)


        self.frame_count += 1

        if self.frame_count == self.tensor.get_base_frame_count():
            self.frame_count = 0

        return self.tensor.get_tensor()
