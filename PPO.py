import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
# https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/blob/master/train.py
################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        # 256 * 240의 이미지를 4개 쌓아서 보내면 
        # input_channels -> 4
        # width     -> 256
        # height    -> 240
        input_channels, width, height = state_dim
        # input_channels = 4
        # width = 256 / 2
        # height = 240 / 2
        
        # input_channels += 12
        layers_info = [
            (64, 3, 2, 1),  # 첫 번째 컨볼루션 레이어: 32채널, 3x3 필터, 스트라이드 2 패딩 1
            (64, 3, 2, 1),  
            (64, 3, 2, 1),
            (64, 3, 2, 1),   
        ]

        # layers_info = [
        #     (32, 3, 2, 1),  # 첫 번째 컨볼루션 레이어: 32채널, 3x3 필터, 스트라이드 2 패딩 1
        #     (32, 3, 2, 1),  
        #     (32, 3, 2, 1),
        #     (32, 3, 2, 1),   
        # ]


        # layers_info = [
        #     (32, 7, 3, 1),  # 첫 번째 컨볼루션 레이어: 32채널, 3x3 필터, 스트라이드 2 패딩 1
        #     (32, 5, 2, 1),  
        #     (64, 5, 2, 1),
        # ]

        # Common feature extractor
        self.actor_features = self._make_layers(input_channels, layers_info)
        self.critic_features = self._make_layers(input_channels, layers_info)
        
        # final_dim = self._calculate_feature_dim(height, width, layers_info)
        final_dim = 2304
        # final_dim = 1152
        # final_dim = 3584
        # print(f"final dim is: {final_dim}")
        # Actor network
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init, device=self.device)
            self.actor = nn.Sequential(
                self.actor_features,
                nn.Linear(final_dim, 512),
                nn.ReLU(),
                nn.Linear(512, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                self.actor_features,
                nn.Linear(final_dim, 512),
                nn.ReLU(),
                nn.Linear(512, action_dim),
                nn.Softmax(dim=-1)
            )


        self.critic = nn.Sequential(
            self.critic_features,
            nn.Linear(final_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )


        
        
    def _make_layers(self, input_channels, layers_info):
        layers = []
        current_channels = input_channels
        for out_channels, kernel_size, stride, padding in layers_info:
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=kernel_size, stride=stride, padding = padding))
            layers.append(nn.ReLU())
            current_channels = out_channels
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def _calculate_feature_dim(self, height, width, layers_info):
        for _, kernel_size, stride, padding in layers_info:
            height = (height - kernel_size) // stride + 1
            width = (width - kernel_size) // stride + 1

        return int(height * width * layers_info[-1][0])  # 마지막 레이어의 채널 수를 곱함
    
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        # print(f"Input state shape: {state.shape}")
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        print("updating parameters")
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)


        

        # 배치 차원 추가를 위해 stack 사용
        # states_with_channel = [torch.unsqueeze(state, 0) for state in self.buffer.states]
        # old_states = torch.stack(states_with_channel, dim=0).to(device)  # 결과는 [batch_size, 1, 120, 128]
        # old_states = torch.stack(self.buffer.states).to(device)
        # old_states = torch.stack(self.buffer.states).to(device)
        old_states = torch.cat(self.buffer.states, dim=0).to(device)
        # convert list to tensor
        # old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        print("updating parameters end")
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       

