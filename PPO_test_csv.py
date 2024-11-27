import os
import glob
import time
import csv
from datetime import datetime
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys

from Game import Game
from PPO import PPO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os


def test():
    print("============================================================================================")
    env_name = "RoboschoolWalker2d-v1"
    has_continuous_action_space = False
    max_ep_len = 16384
    action_std = 0.1
    render = True
    frame_delay = 0
    total_test_episodes = 1
    K_epochs = 10
    eps_clip = 0.05
    gamma = 0.95
    lr_actor = 0.000003
    lr_critic = 0.000003

    env = Game(256,240, visualize=True)
    env.init_tensor_size(84, 84)
    state_dim = [4,84,84]
    action_dim = 12

    K_epochss = [5, 10, 15, 20]
    eps_clips = [0.05, 0.1, 0.15, 0.2, 0.25]
    gammas = [0.7,0.8,0.9,0.95,0.99]



    ppo_agents = []
    results = []

    for K_epochs in K_epochss:
        for eps_clip in eps_clips:
            for gamma in gammas:
                ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
                ppo_agents.append((ppo_agent, K_epochs, eps_clip, gamma))

    checkpoint_path = "./PPO_preTrained/SuperMarioBros-Nes/56_5500000.pth"
    print("loading network from : " + checkpoint_path)

    # CSV 파일 생성 및 헤더 작성
    with open('test_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['K_epochs', 'eps_clip', 'gamma', 'episode', 'reward', 'log_running_trial', 'log_world_cleared', 'world_clear_rate'])
        total_test_num = len(K_epochss) * len(eps_clips) * len(gammas)
        i = 0
        for ppo_agent, K_epochs, eps_clip, gamma in ppo_agents:
            i += 1
            ppo_agent.load(checkpoint_path)
            for ep in range(1, total_test_episodes+1):
                ep_reward, log_running_trial, log_world_cleared = run_episode(env, ppo_agent, max_ep_len, render, frame_delay)
                if log_running_trial > 0:
                    world_clear_rate = log_world_cleared / log_running_trial
                else:
                    world_clear_rate = 0
                writer.writerow([K_epochs, eps_clip, gamma, ep, ep_reward, log_running_trial, log_world_cleared, world_clear_rate])
            print(f"{i / total_test_num * 100: .3f}% clear")
def run_episode(env, ppo_agent, max_ep_len, render, frame_delay):
    state = env.reset()
    ep_reward = 0
    log_running_trial = 0
    log_world_cleared = 0
    states = []
    states.append(state)
    prev_action = 0

    for t in range(1, max_ep_len+1):
        if t % 4 != 0:
            state, reward, done, is_dead = env.step_new_ppo(prev_action)
            states.append(state)
            time.sleep(frame_delay)
            if is_dead:
                log_running_trial += 1 
            if done:
                log_world_cleared += 1
            continue
        
        state_cat = torch.cat(states, dim=1)
        action = ppo_agent.select_action(state_cat)
        
        states = []
        state, reward, done, is_dead = env.step_new_ppo(action)
        if is_dead:
            log_running_trial += 1 
        if done:
            log_world_cleared += 1
        prev_action = action
        states.append(state)
        ep_reward += reward
        if render:
            time.sleep(frame_delay)
        # if done:
        #     break

    ppo_agent.buffer.clear()
    return ep_reward, log_running_trial, log_world_cleared




def plot_combined():
    # 실제 파일 경로로 수정
    csv_file_path = 'test_results.csv'
    
    df = pd.read_csv(csv_file_path)

    # eps_clip과 gamma에 따라 그룹화하고 각 그룹의 평균을 계산하여 K_epochs를 무시
    averaged_df = df.groupby(['eps_clip', 'gamma']).agg({
        'reward': 'mean',
        'log_running_trial': 'mean',
        'log_world_cleared': 'mean'
    }).reset_index()
    
    # World Clear Rate 계산
    averaged_df['world_clear_rate'] = averaged_df['log_world_cleared'] / averaged_df['log_running_trial']
    
    # 시각화 준비
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # Reward plot
    for key, grp in averaged_df.groupby(['eps_clip']):
        ax[0].plot(grp['gamma'], grp['reward'], marker='o', label=f'eps_clip={key}')
    ax[0].set_title('Combined Average Reward vs Gamma')
    ax[0].set_xlabel('Gamma')
    ax[0].set_ylabel('Average Reward')
    ax[0].legend(title='eps_clip', bbox_to_anchor=(1.05, 1), loc='upper left')

    # World Clear Rate plot
    for key, grp in averaged_df.groupby(['eps_clip']):
        ax[1].plot(grp['gamma'], grp['world_clear_rate'], marker='o', label=f'eps_clip={key}')
    ax[1].set_title('Combined Average World Clear Rate vs Gamma')
    ax[1].set_xlabel('Gamma')
    ax[1].set_ylabel('World Clear Rate')
    ax[1].legend(title='eps_clip', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()





def analyze_impact():
    # 실제 파일 경로로 수정
    csv_file_path = 'test_results.csv'
    df = pd.read_csv(csv_file_path)

    # 데이터 전처리: World Clear Rate 계산
    df['world_clear_rate'] = df['log_world_cleared'] / df['log_running_trial'].replace(0, 1)  # 0으로 나누는 것을 방지

    # 상관 관계 분석
    correlation_matrix = df[['gamma', 'eps_clip', 'reward', 'world_clear_rate']].corr()
    print("Correlation matrix:")
    print(correlation_matrix)

    # 결과 저장 폴더 생성
    result_folder = './results_csv'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # 상관관계 행렬 저장
    correlation_matrix.to_csv(f'{result_folder}/correlation_matrix.csv')

    # 선형 회귀 모델을 사용하여 영향력 분석
    X = df[['gamma', 'eps_clip']]
    y_reward = df['reward']
    y_clear_rate = df['world_clear_rate']

    X = sm.add_constant(X)  # 상수항 추가
    model_reward = sm.OLS(y_reward, X).fit()
    model_clear_rate = sm.OLS(y_clear_rate, X).fit()

    # 회귀 결과 저장
    with open(f'{result_folder}/linear_regression_reward.txt', 'w') as file:
        file.write(model_reward.summary().as_text())
    with open(f'{result_folder}/linear_regression_clear_rate.txt', 'w') as file:
        file.write(model_clear_rate.summary().as_text())

    print("Linear Regression Result for Reward:")
    print(model_reward.summary())
    print("Linear Regression Result for World Clear Rate:")
    print(model_clear_rate.summary())

    # Reward vs Gamma for different eps_clips
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='gamma', y='reward', hue='eps_clip', marker='o')
    plt.title('Reward vs Gamma for different eps_clips')
    plt.xlabel('Gamma')
    plt.ylabel('Reward')
    plt.legend(title='eps_clip')
    plt.savefig(f'{result_folder}/reward_vs_gamma.png')
    plt.show()

    # World Clear Rate vs Gamma for different eps_clips
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='gamma', y='world_clear_rate', hue='eps_clip', marker='o')
    plt.title('World Clear Rate vs Gamma for different eps_clips')
    plt.xlabel('Gamma')
    plt.ylabel('World Clear Rate')
    plt.legend(title='eps_clip')
    plt.savefig(f'{result_folder}/clear_rate_vs_gamma.png')
    plt.show()










if __name__ == '__main__':
    app = QApplication(sys.argv)
    # test()
    # analyze_impact()
    plot_combined()
