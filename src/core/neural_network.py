import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
from typing import List, Tuple
import random

class BioNeuralNetwork(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 8, output_size: int = 3):
        super(BioNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
        self.memory = deque(maxlen=1000)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """存储经验到记忆库"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int = 32):
        """从记忆库中学习"""
        if len(self.memory) < batch_size:
            return

        # 随机采样一批经验
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # 转换为tensor
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 计算当前Q值
        current_q_values = self.forward(states)
        next_q_values = self.forward(next_states)

        # 计算目标Q值
        target_q_values = current_q_values.clone()
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + 0.95 * torch.max(next_q_values[i])

        # 训练网络
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def transfer_knowledge(self, offspring: 'BioNeuralNetwork', noise_level: float = 0.1):
        """将知识传授给后代，并添加随机噪声"""
        for param, offspring_param in zip(self.parameters(), offspring.parameters()):
            noise = torch.randn_like(param) * noise_level
            offspring_param.data.copy_(param.data + noise)

    def get_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """选择动作（ε-贪婪策略）"""
        if np.random.random() < epsilon:
            return np.random.randint(3)  # 随机选择动作
        with torch.no_grad():
            state = torch.FloatTensor(state)
            q_values = self.forward(state)
            return torch.argmax(q_values).item()

class BioRL(nn.Module):
    def __init__(self, input_size):
        super(BioRL, self).__init__()
        self.input_size = input_size
        self.hidden_size = 64
        self.output_size = 8  # 8个可能的动作
        
        # 使用简单的前馈神经网络
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
        
        # 将所有层移动到CPU
        self.to('cpu')
        
        # 经验回放缓冲区
        self.memory = []
        self.max_memory = 1000
        self.batch_size = 32
        
        # 学习参数
        self.learning_rate = 0.001
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 0.1  # 探索率
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def forward(self, x):
        # 确保输入在CPU上
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        x = x.to('cpu')
        
        # 如果输入是一维的，添加批次维度
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def remember(self, state, action, reward, next_state):
        # 将经验添加到记忆中
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
            
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        # 从记忆中随机采样
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([experience[0] for experience in batch])).to('cpu')
        actions = torch.LongTensor(np.array([experience[1] for experience in batch])).to('cpu')
        rewards = torch.FloatTensor(np.array([experience[2] for experience in batch])).to('cpu')
        next_states = torch.FloatTensor(np.array([experience[3] for experience in batch])).to('cpu')
        
        # 计算当前Q值和目标Q值
        current_q_values = self.forward(states)
        next_q_values = self.forward(next_states)
        
        target_q_values = current_q_values.clone()
        for i in range(self.batch_size):
            target_q_values[i][actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i])
            
        # 更新网络
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def get_action(self, state):
        # ε-贪婪策略
        if random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)
            
        with torch.no_grad():
            q_values = self.forward(state)
            return torch.argmax(q_values).item()
            
    def transfer_knowledge(self, other_brain):
        """从其他大脑转移知识"""
        # 确保两个网络在同一个设备上
        other_brain.to('cpu')
        
        # 复制参数
        self.load_state_dict(other_brain.state_dict())
        
        # 稍微改变参数以产生变异
        with torch.no_grad():
            for param in self.parameters():
                mutation = torch.randn_like(param) * 0.1
                param.add_(mutation)