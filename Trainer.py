import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import deque
import numpy as np
import os
import threading
import queue

from Game import Game
from network import PPOAgent
from Yolo_Model import Yolo_Model
from InputType import AgentInput

# 하이퍼파라미터
NUM_EPISODES = 1000
MAX_STEPS = 10000 #30* 401 # 한 프레임30, 총 400초
GAMMA = 0.90
GAE_LAMBDA = 0.95

BATCH_SIZE = 64
CLIP_EPSILON = 0.15  # PPO 클리핑 epsilon 값
LR_POLICY = 1e-4  # 학습률
LR_VALUE = 1e-4
actualSteps = MAX_STEPS / 4
UPDATE_INTERVAL = actualSteps / 10  # 업데이트 주기, 에피소드 하나에 10번 업데이트

class Trainer():
    def __init__(self, agent: PPOAgent, game: Game, use_yolo = False):
        # 에이전트 및 모델 초기화
        input_dim = 3864  # YOLO state + Mario state + 이전 행동
        self.action_dim = 12  # 12차원 행동 공간
        hidden_dims = [1024, 256]  # 네트워크 hidden layer 크기

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = agent
        # self.agent = PPOAgent(input_dim = input_dim, hidden_dims = hidden_dims, output_dim = self.action_dim, lr_policy = LR_POLICY, lr_value = LR_VALUE)  # PPO 에이전트
        # self.agent.to(self.device)
        self.AgentInput = AgentInput(self.agent)
        self.yolo_model = Yolo_Model(x_pixel_num=256, y_pixel_num=240)  # YOLO 모델
        self.game = game  # 게임 환경
        self.games = [] # 멀티스레딩에 이용

        # 옵티마이저 설정
        # self.optimizer = optim.Adam(self.agent.parameters(), lr=LR)
        self.use_yolo = use_yolo


        # 0부터 순서대로 
        # null, 아래(숙이기), 좌, 우, a=jump, b=달리기 or 공격, 우 + 점프, 좌 + 점프, 우 + 공격, 좌 + 공격, 우 + 공격 + 점프, 좌 + 공격 + 점프
        # 0 ~ 11의 정수
        self.prev_action = 0 

        ### 저장
        self.model_dir = './models'  # 모델을 저장할 디렉터리
        os.makedirs(self.model_dir, exist_ok=True)


        ###### 멀티스레딩 사용 변수 #########
        self.threading_num = None
        self.games = []
        self.data_queue = queue.Queue()
        self.update_lock = threading.Lock()
        self.training = True
        self.prev_actions = None

    # def save_model(self, episode):
    #     """ 모델의 파라미터를 저장합니다. """
    #     current_time = time.time()
    #     filename = os.path.join(self.model_dir, f"ppo_agent_episode_{episode}_{current_time}.pth")
    #     torch.save(self.agent.state_dict(), filename)
    #     print(f"Model saved to {filename} at episode {episode}")

    def save_model(self, episode, avg_reward, avg_policy_loss, avg_value_loss):
        print("save_model")

        """ 모델의 파라미터와 하이퍼파라미터, 성능 지표를 저장합니다. """
        current_time = time.time()
        filename = os.path.join(self.model_dir, f"ppo_agent_episode_{episode}_{int(current_time)}.pth")
        
        # 저장할 정보를 딕셔너리 형태로 구성
        model_info = {
            'episode': episode,
            'model_state_dict': self.agent.state_dict(),
            'hyperparameters': {
                'gamma': GAMMA,
                'gae_lambda': GAE_LAMBDA,
                'batch_size': BATCH_SIZE,
                'clip_epsilon': CLIP_EPSILON,
                'lr_policy': LR_POLICY,
                'lr_value': LR_VALUE,
                'update_interval': UPDATE_INTERVAL
            },
            'performance_metrics': {
                'average_reward': avg_reward,
                'avg_policy_loss': avg_policy_loss,
                'avg_value_loss': avg_value_loss,

            }
        }

        torch.save(model_info, filename)
        print(f"Model and metrics saved to {filename} at episode {episode}")


    def get_tensor(self):
        if self.use_yolo:
            mario_state = self.game.get_mario_state()  # 4프레임 동안의 마리오 상태 (12차원 벡터)
            # YOLO 모델을 사용하여 상태 추출
            yolo_input_img = self.game.get_yolo_input_img()
            tensor_state = self.yolo_model.get_tensor(yolo_input_img, mario_state)
            
            return tensor_state
        else:
            tensor_state = self.game.get_tensor()
            
            # print(tensor_state)
            return tensor_state
        
    def get_tensor_by_given_game(self, game: Game):
        if self.use_yolo:
            mario_state = game.get_mario_state()  # 4프레임 동안의 마리오 상태 (12차원 벡터)
            # YOLO 모델을 사용하여 상태 추출
            yolo_input_img = game.get_yolo_input_img()
            tensor_state = self.yolo_model.get_tensor(yolo_input_img, mario_state)
            
            return tensor_state
        else:
            tensor_state = game.get_tensor()
            
            # print(tensor_state)
            return tensor_state

    def train_test(self):
        action = np.array([0] * 9)
        action[8] = 1
        for i in range(10000000):
            reward, done, _ = self.game.step(action)
            time.sleep(1/60)


    def train(self):
        for episode in range(NUM_EPISODES):
            state = None  # 초기 상태
            states = []
            rewards, log_probs, values, masks, actions = [], [], [], [], []
            old_log_probs, old_values = [], []  # 이전 log_prob와 value를 저장
            print(f"episode {episode} start")
            start_time = time.time()
            actual_steps = 0

            avg_policy_loss = 0
            avg_value_loss = 0
            avg_reward = 0
            for step in range(MAX_STEPS):
                current_time = time.time()

                   
                tensor_state = self.get_tensor()
                # 누적 프레임이 충분하지 않으면 이전 action을 입력
                if tensor_state is None:
                    action_np = self.AgentInput.get_action_np(self.prev_action)
                    reward, done, _ = self.game.step(action_np)
                    continue


                actual_steps += 1
                # 이전 action을 one-hot 벡터로 결합
                # prev_action = self.game.get_prev_action_index()  # 12차원 벡터 (이전 행동)
                prev_action_one_hot = torch.zeros(self.action_dim)
                prev_action_one_hot[self.prev_action] = 1
                
                # # 마리오 상태 12차원 벡터 (이미 4프레임 동안의 정보가 제공됨)
                # mario_state_tensor = torch.tensor(mario_state, dtype=torch.float32)

                # 입력 벡터를 결합
                # full_state = torch.cat([tensor_state, mario_state_tensor, prev_action_one_hot])

                full_state = torch.cat([tensor_state, prev_action_one_hot])
                full_state = full_state.to(self.device)

                # 에이전트 행동 선택
                
                action_int, action_tensor, log_prob, value = self.agent.select_action(full_state)
                self.prev_action = action_int
                action_np = self.AgentInput.get_action_np(action_int) # action = np.array([0] * 9)
                
                action_one_hot = torch.zeros(self.action_dim)
                action_one_hot[action_int] = 1
                # 보상 및 게임 정보 업데이트
                reward, done, _ = self.game.step(action_np)  # step() 메소드에서 보상과 종료 여부 받기
                
                
                if actual_steps % 1000 == 0:
                    print(f"episode {episode} {step / MAX_STEPS:.2f}%")
                    print(f"current_step: {step}")
                    print(f"prev action: {self.prev_action}")
                    print(f"elapsed time: {current_time - start_time}")
                    print(f"reward: {reward}")
                    print(f"log_prob: {log_prob}")
                
                # print(f"reward: {reward}")
                # print(f"action: {action_int}")
                # log_prob = 
                states.append(full_state)  # 현재 상태를 states 리스트에 추가
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                masks.append(1 - done)
                actions.append(action_one_hot)
                old_log_probs.append(log_prob.detach())  # 이전 log_prob 저장
                old_values.append(value.detach())  # 이전 value 저장
                # 에피소드 종료 조건
                if done:
                    break

                # 업데이트 주기마다 에이전트를 학습
                if actual_steps % UPDATE_INTERVAL == 0:
                    print('update interval')
                    if not log_probs:  # log_probs 리스트가 비어 있으면 업데이트를 스킵
                        print("Skipping update: 'log_probs' list is empty.")
                        continue

                    # 모든 리스트의 길이가 동일한지 확인
                    if not all(len(lst) == len(log_probs) for lst in [rewards, values, masks, old_log_probs, old_values]):
                        print("Skipping update: Not all lists are of equal length.")
                        continue

                    # 리스트 내 요소의 차원 확인
                    if any(lp.dim() == 0 for lp in log_probs):  # log_probs의 각 요소가 0차원인지 확인
                        print("Skipping update: 'log_probs' contains zero-dimensional tensors.")
                        continue
                    returns, advantages = self.compute_gae(rewards, values, masks)
                    
                    # PPO 업데이트: 클리핑 기법을 사용하여 정책 업데이트
                    avg_policy_loss, avg_value_loss = self.agent.update(states, actions, returns, advantages, log_probs, old_log_probs, old_values, BATCH_SIZE, CLIP_EPSILON)
                    avg_reward = np.mean(rewards)
                    
                    # 매 업데이트 후 로그 초기화
                    rewards, log_probs, values, masks, actions = [], [], [], [], []
                    old_log_probs, old_values = [], []
                    states = []
            
            ## 에피소드가 끝날때마다 모델 저장
            self.save_model(episode, avg_reward, avg_policy_loss, avg_value_loss)
            print(f"Episode {episode + 1}/{NUM_EPISODES} completed.")



    def train_multithreading(self, visualize, num_games, x_pixel_num = 256, y_pixel_num = 240):
        self.games = [Game(x_pixel_num, y_pixel_num, visualize) for _ in range(num_games)]
        self.prev_actions = {}
        for i in range(num_games):
            self.prev_actions[i] = 0

        threads = [
            threading.Thread(target=self.agent_run, args=(game_id, game, self.agent))
            for game_id, game in enumerate(self.games)
        ]
        for thread in threads:
            thread.start()
        
        updater_thread = threading.Thread(target=self.update_network)
        updater_thread.start()

        for thread in threads:
            thread.join()
        
        updater_thread.join()     
        return

    def agent_run(self, game_id, game, agent):
        episode = 0
        while self.training:
            state = None
            states, actions, rewards, values, log_probs, masks = [], [], [], [], [], []
            old_log_probs, old_values = [], [] 

            actual_steps = 0
            avg_policy_loss = 0
            avg_value_loss = 0
            avg_reward = 0

            for step in range(MAX_STEPS):
                tensor_state = self.get_tensor_by_given_game(game)
                prev_action = self.prev_action[game_id]
                if tensor_state is None:
                    action_np = self.AgentInput.get_action_np(prev_action)
                    reward, done, _ = game.step(action_np)
                    continue

                actual_steps += 1
                # 이전 action을 one-hot 벡터로 결합
                # prev_action = self.game.get_prev_action_index()  # 12차원 벡터 (이전 행동)
                prev_action_one_hot = torch.zeros(self.action_dim)
                prev_action_one_hot[prev_action] = 1

                full_state = torch.cat([tensor_state, prev_action_one_hot])
                full_state = full_state.to(self.device)


                # 에이전트 행동 선택
                action_int, action_tensor, log_prob, value = self.agent.select_action(full_state)
                self.prev_action = action_int
                action_np = self.AgentInput.get_action_np(action_int) # action = np.array([0] * 9)
                
                action_one_hot = torch.zeros(self.action_dim)
                action_one_hot[action_int] = 1
                # 보상 및 게임 정보 업데이트
                reward, done, _ = self.game.step(action_np)  # step() 메소드에서 보상과 종료 여부 받기
                
                if actual_steps % 1000 == 0:
                    print(f"episode {episode} {step / MAX_STEPS:.2f}%")
                    print(f"current_step: {step}")
                    print(f"prev action: {self.prev_action}")
                    # print(f"elapsed time: {current_time - start_time}")
                    print(f"reward: {reward}")
                    print(f"log_prob: {log_prob}")
                

                states.append(full_state)  # 현재 상태를 states 리스트에 추가
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                masks.append(1 - done)
                actions.append(action_one_hot)
                old_log_probs.append(log_prob.detach())  # 이전 log_prob 저장
                old_values.append(value.detach())  # 이전 value 저장


                if done:
                    game.reset()
                    break

                if step % UPDATE_INTERVAL == 0:
                    with self.update_lock:
                        self.data_queue.put((states, actions, rewards, values, log_probs, masks, old_log_probs, old_values))
                    states, actions, rewards, values, log_probs, masks = [], [], [], [], [], []
                    old_log_probs, old_values = [], []

            episode += 1
            if episode >= NUM_EPISODES:
                self.training = False

    def update_network(self):
        while self.training or not self.data_queue.empty():
            all_states, all_actions, all_rewards, all_values, all_log_probs, all_masks, all_old_log_probs, all_old_values = [], [], [], [], [], [], [], []
            while not self.data_queue.empty():
                data = self.data_queue.get()
                states, actions, rewards, values, log_probs, masks, old_log_probs, old_values = data
                all_states.extend(states)
                all_actions.extend(actions)
                all_rewards.extend(rewards)
                all_values.extend(values)
                all_log_probs.extend(log_probs)
                all_masks.extend(masks)
                all_old_log_probs.extend(old_log_probs)
                all_old_values.extend(old_values)

            if all_states:
                # 신경망 업데이트 로직
                all_returns, all_advantages = self.compute_gae(all_rewards, all_values, all_masks)
                # self.global_agent.update(all_states, all_actions, all_rewards, all_values, all_log_probs, all_masks, BATCH_SIZE, CLIP_EPSILON)
                avg_policy_loss, avg_value_loss = self.agent.update(all_states, all_actions, all_returns, all_advantages, all_log_probs, all_old_log_probs, all_old_values, BATCH_SIZE, CLIP_EPSILON)
                avg_reward = np.mean(all_rewards)
            
    def compute_gae(self, rewards, values, masks, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
        # GAE advantage 계산
        values = values + [0]  # 마지막 상태 값 추가
        returns, advantages = [], []
        gae = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + gamma * gae_lambda * masks[i] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        return returns, advantages



