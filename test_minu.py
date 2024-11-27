import multiprocessing
from Game import Game
from PyQt5.QtWidgets import QApplication
import sys
from Visualizer import GameFrameVisualizer, GridWindow
import pygame
import os
import retro

def run_game(index, x_pixel_num, y_pixel_num, visualize, window_x, window_y):
    app = QApplication(sys.argv)
    # os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"
    # pygame.init()
    # print(window_x)
    game = Game(x_pixel_num, y_pixel_num, visualize, window_x = window_x, window_y = window_y, center = True)
    # grid_window = GridWindow()
    print(f"Game {index} is running")
    # 게임 로직 실행
    game.run()
    # game.close()  # 게임 종료 시 반드시 close 호출

# def run_game(index, x_pixel_num, y_pixel_num, visualize, window_x, window_y):
#     # 각 프로세스에서 개별적으로 os.environ 설정
#     os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"
#     pygame.init()

#     env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
#     env.reset()

#     if visualize:
#         screen = pygame.display.set_mode((x_pixel_num, y_pixel_num))
#         clock = pygame.time.Clock()

#     while True:
#         obs, reward, done, info = env.step(env.action_space.sample())
#         if visualize:
#             frame = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
#             screen.blit(frame, (0, 0))
#             pygame.display.flip()
#             clock.tick(60)
#         if done:
#             env.reset()

def multithread_test():
    pygame.init()
    app = QApplication([])

    num_games = 3
    x_pixel_num, y_pixel_num, visualize = 256, 240, True
    screen_width = 1920 * 2  # 화면 너비 (예제)
    screen_height = 1080 * 2  # 화면 높이 (예제)

    gap = 300  # 창들 간의 간격
    start_x_offset = 1000  # 시작 위치 x 오프셋
    start_y_offset = 1000  # 시작 위치 y 오프셋
    x_windows = [
        start_x_offset + (i * (x_pixel_num + gap)) % screen_width
        for i in range(num_games)
    ]
    y_windows = [
        start_y_offset
        + (i * (y_pixel_num + gap))
        // (screen_width // (x_pixel_num + gap))
        * (y_pixel_num + gap)
        for i in range(num_games)
    ]

    processes = []
    for i in range(num_games):
        process = multiprocessing.Process(
            target=run_game, args=(i, x_pixel_num, y_pixel_num, visualize, x_windows[i], y_windows[i])
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


def tensor_test():
    import torch

    # 예제의 이전 행동 (prev_action), action_dim, width, height
    prev_action = 2
    action_dim = 12
    width = 84
    height = 84

    # 이전 행동을 원-핫 인코딩하고 적절한 크기로 확장
    prev_action_one_hot = torch.nn.functional.one_hot(torch.tensor(prev_action), num_classes=action_dim)
    prev_action_one_hot = prev_action_one_hot.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, action_dim, 1, 1] 크기
    prev_action_one_hot = prev_action_one_hot.expand(-1, -1, width, height)  # 상태 텐서와 동일한 공간 차원으로 확장

    print(prev_action_one_hot.shape)
    print(prev_action_one_hot)


if __name__ == "__main__":
    # game = Game(256,240,True)
    tensor_test()
    
