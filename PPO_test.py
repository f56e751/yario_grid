import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
# import roboschool
from Game import Game

from PPO import PPO
from PyQt5.QtWidgets import QApplication
import sys
import cv2

from PIL import Image
def save_video_mp4(frames, filename, fps=30):
    # 첫 프레임으로부터 비디오 해상도 추출
    if not frames:
        print("No frames to process.")
        return

    first_frame = np.array(frames[0])
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), True)

    if not out.isOpened():
        print("Video writer could not be initialized. Check the codec and file path.")
        return

    for pil_image in frames:
        # Convert PIL Image to an OpenCV-compatible numpy array
        frame = np.array(pil_image)  # Convert PIL image to numpy array
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        
        # Write the frame
        out.write(frame)

    out.release()
    print("Video saved successfully.")

#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    env_name = "RoboschoolWalker2d-v1"
    has_continuous_action_space = False
    max_ep_len = 4096           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 1/240             # if required; add delay b/w frames

    total_test_episodes = 1    # total num of testing episodes

    K_epochs = 10               # update policy for K epochs
    eps_clip = 0.1              # clip parameter for PPO
    gamma = 0.95                  # discount factor

    lr_actor = 0.000003           # learning rate for actor
    lr_critic = 0.000003           # learning rate for critic

    #####################################################

    env = Game(256,240, visualize=True)
    env.init_tensor_size(84, 84)

    # state space dimension
    state_dim = [4,84,84]

    # action space dimension
    # if has_continuous_action_space:
    #     action_dim = env.action_space.shape[0]
    # else:
    #     action_dim = env.action_space.n
    
    action_dim = 12
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # K_epochss = [5, 10, 15, 20]
    # eps_clips = [0.05, 0.1, 0.15, 0.2, 0.25]
    # gammas = [0.7,0.8,0.9,0.95,0.99]
    
    # ppo_agents = []
    # for K_epochs in K_epochss:
    #     for eps_clip in eps_clips:
    #         for gamma in gammas:
    #             ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    #             ppo_agents.append(ppo_agent)



    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    # directory = "PPO_preTrained" + '/' + env_name + '/'
    # checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    checkpoint_path = "./PPO_preTrained/SuperMarioBros-Nes/45_100000.pth"
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    frame_stack_num = 4
    prev_action = 0
    state = env.reset()
    
    # with open('test_results.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['K_epochs', 'eps_clip', 'gamma', 'episode', 'reward', 'log_running_trial', 'log_world_cleared', 'world_clear_rate'])

    frames = []  # 영상을 저장할 프레임 리스트
    for ep in range(1, total_test_episodes+1):
    # for ppo_agent in ppo_agents:
        ep_reward = 0
        ppo_agent.load(checkpoint_path)
        log_world_cleared = 0
        log_running_trial = 0
        states = []
        states.append(state)
        for t in range(1, max_ep_len+1):
            # print(t)
            if t % frame_stack_num != 0:
                # print(t % frame_stack_num)
                state, reward, done, is_dead = env.step_new_ppo(prev_action)
                states.append(state)
                frame = env.get_yolo_input_img()
                frames.append(frame)
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


            frame = env.get_yolo_input_img()
            frames.append(frame)
            ep_reward += reward
            if render:
                # env.render()
                time.sleep(frame_delay)

            # if done:
            #     break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0


    filename = f"./videos/{45_100000}.mp4"
    save_video_mp4(frames, filename, 60)
    # env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")




if __name__ == '__main__':
    app = QApplication(sys.argv)
    test()