import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PPO 에이전트 네트워크 정의
class PPOAgent(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, lr_policy=1e-3, lr_value=1e-3):
        super(PPOAgent, self).__init__()
        self.input_dim = input_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ########### action head ###########
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.action_head = nn.Linear(hidden_dims[1], output_dim)
        

        ############ value head ###########
        self.fc1_val = nn.Linear(input_dim, hidden_dims[0])
        self.value_head = nn.Linear(hidden_dims[0], 1)

        ############# optimizer ############
        self.policy_optimizer = optim.Adam(self.get_policy_parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.get_value_parameters(), lr=lr_value)

        # 파라미터 반환 메소드
    def get_policy_parameters(self):
        return list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.action_head.parameters())

    def get_value_parameters(self):
        return list(self.fc1_val.parameters()) + list(self.value_head.parameters())

    def forward(self, x):
        x_val = F.relu(self.fc1_val(x))
        value = self.value_head(x_val)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        
        return action_logits, value

    def select_action(self, state):
        # 입력 상태의 크기가 input_dim과 일치하는지 검사
        if state.size(-1) != self.input_dim:
            raise ValueError(f"Expected input dimension is {self.input_dim}, but got {state.size(-1)}")
        
        action_logits, value = self.forward(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        # return action.item(), dist.log_prob(action), value
        return action.item(), action, dist.log_prob(action).unsqueeze(0), value
    

    def update(self, states, actions, returns, advantages, log_probs, old_log_probs, old_values, batch_size, clip_epsilon):
        # Tensor로 변환하는 과정, 모든 텐서를 self.device로 이동
        states = torch.stack(states, dim=0).to(self.device).detach()
        actions = torch.cat(actions).to(self.device).detach()  # 행동 인덱스 텐서
        returns = torch.cat(returns).to(self.device).detach()
        advantages = torch.cat(advantages).to(self.device).detach()
        log_probs = torch.cat(log_probs).to(self.device)
        old_log_probs = torch.cat(old_log_probs).to(self.device).detach()
        old_values = torch.cat(old_values).to(self.device).detach()
  
        policy_losses = []
        value_losses = []

        # 정책 및 가치 네트워크의 손실을 계산
        for _ in range(10):  # PPO는 일반적으로 여러 에폭동안 같은 샘플로 업데이트를 수행
            indices = torch.randperm(len(states)).to(states.device)  # 데이터를 셔플
            for i in range(0, len(states), batch_size):  # 미니 배치 학습
                sampled_indices = indices[i:i + batch_size]
                sampled_states = states[sampled_indices]
                sampled_actions = actions[sampled_indices]
                sampled_returns = returns[sampled_indices]
                sampled_advantages = advantages[sampled_indices]
                sampled_log_probs = log_probs[sampled_indices]
                sampled_old_log_probs = old_log_probs[sampled_indices]
                sampled_old_values = old_values[sampled_indices]

                # 새로운 log_prob, values 계산
                action_logits, values = self.forward(sampled_states)
                new_log_probs = torch.log(torch.softmax(action_logits, dim=-1) + 1e-10)  # log_softmax와 동일


                # 샘플링된 행동의 로그 확률 추출
                # sampled_actions = sampled_actions.unsqueeze(1)  # 인덱스를 위해 차원 추가
                sampled_actions = sampled_actions.to(torch.int64).unsqueeze(1)
                sampled_new_log_probs = torch.gather(new_log_probs, 1, sampled_actions).squeeze(1)  # 적절한 로그 확률 추출
                
                
                # 확률비 계산
                ratios = torch.exp(sampled_new_log_probs - sampled_old_log_probs)

                # PPO 클리핑된 목적 함수
                surr1 = ratios * sampled_advantages
                surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * sampled_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 가치 손실
                # value_loss = F.mse_loss(values, sampled_returns)
                value_loss = F.mse_loss(values.squeeze(1), sampled_returns)

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                # # 전체 손실
                # loss = policy_loss + 0.5 * value_loss

                # # 업데이트
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
        avg_policy_loss = sum(policy_losses) / len(policy_losses)
        avg_value_loss = sum(value_losses) / len(value_losses)
        return avg_policy_loss, avg_value_loss



class PPOAgent_CNN(nn.Module):
    def __init__(self, input_channels, feature_dim, output_dim, lr_policy=1e-3, lr_value=1e-3):
        super(PPOAgent, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ########### CNN layers ###########
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * feature_dim * feature_dim, 512),  # Adjust the size according to your final feature map dimensions
            nn.ReLU()
        )

        ########### action head ###########
        self.action_head = nn.Linear(512, output_dim)

        ############ value head ###########
        self.value_head = nn.Linear(512, 1)

        ############# optimizers ############
        self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': lr_critic}
            ])
        # self.policy_optimizer = optim.Adam(self.get_policy_parameters(), lr=lr_policy)
        # self.value_optimizer = optim.Adam(self.get_value_parameters(), lr=lr_value)

    def get_policy_parameters(self):
        return list(self.features.parameters()) + list(self.action_head.parameters())

    def get_value_parameters(self):
        return list(self.features.parameters()) + list(self.value_head.parameters())

    def forward(self, x):
        x = self.features(x)
        action_logits = self.action_head(x)
        value = self.value_head(x)
        return action_logits, value

    def select_action(self, state):
        if state.size(1) != self.features[0].in_channels:
            raise ValueError(f"Expected input channel is {self.features[0].in_channels}, but got {state.size(1)}")
        
        state = state.to(self.device)
        action_logits, value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).unsqueeze(0), value

    def update(self, states, actions, returns, advantages, log_probs, old_log_probs, old_values, batch_size, clip_epsilon):
        # 각각의 tensor를 GPU로 이동
        states = torch.stack(states).to(self.device).detach()
        actions = torch.cat(actions).to(self.device).detach()
        returns = torch.cat(returns).to(self.device).detach()
        advantages = torch.cat(advantages).to(self.device).detach()
        log_probs = torch.cat(log_probs).to(self.device)
        old_log_probs = torch.cat(old_log_probs).to(self.device).detach()
        old_values = torch.cat(old_values).to(self.device).detach()

        policy_losses = []
        value_losses = []

        for _ in range(10):
            indices = torch.randperm(len(states)).to(states.device)
            for i in range(0, len(states), batch_size):
                sampled_indices = indices[i:i+batch_size]
                sampled_states = states[sampled_indices]
                sampled_actions = actions[sampled_indices]
                sampled_returns = returns[sampled_indices]
                sampled_advantages = advantages[sampled_indices]
                sampled_log_probs = log_probs[sampled_indices]
                sampled_old_log_probs = old_log_probs[sampled_indices]
                sampled_old_values = old_values[sampled_indices]

                # Forward pass
                action_logits, values = self.forward(sampled_states)
                new_log_probs = torch.log(F.softmax(action_logits, dim=-1) + 1e-10)

                sampled_actions = sampled_actions.to(torch.int64).unsqueeze(1)
                sampled_new_log_probs = torch.gather(new_log_probs, 1, sampled_actions).squeeze(1)

                # PPO objective function
                ratios = torch.exp(sampled_new_log_probs - sampled_old_log_probs)
                surr1 = ratios * sampled_advantages
                surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * sampled_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(1), sampled_returns)

                # Optimizers
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        return sum(policy_losses) / len(policy_losses), sum(value_losses) / len(value_losses)



# 네트워크 초기화 함수
def create_agent(input_dim, hidden_dims, output_dim):
    agent = PPOAgent(input_dim, hidden_dims, output_dim).to(device)
    return agent

def test_agent():
    # 에이전트 생성
    input_dim = 24  # 예시로 24차원 상태 벡터를 가정
    hidden_dims = [128, 64]  # 은닉층 차원 설정
    output_dim = 4  # 예를 들어, 4개의 가능한 행동을 가정

    agent = create_agent(input_dim, hidden_dims, output_dim)

    for i in range(50):
        # 더미 입력 데이터 생성 (랜덤)
        dummy_state = torch.rand((1, input_dim)).to(device)

        # 행동 선택 및 출력
        action, action_tensor, log_prob, value = agent.select_action(dummy_state)
        print("Selected Action:", action)
        # print("Log Probability of Selected Action:", log_prob)
        # print("Value of the current state:", value)






if __name__ == "__main__":
    # GPU 장치 설정 (GPU가 있으면 GPU 사용, 없으면 CPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_agent()

