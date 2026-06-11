import torch
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):
    """策略网络（Actor）"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = torch.nn.LayerNorm(hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.dropout = torch.nn.Dropout(p=0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.ln1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.ln2(x)
        x = self.dropout(x)
        return torch.tanh(self.fc3(x))


class ValueNet(torch.nn.Module):
    """价值网络（Critic）"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ValueNet, self).__init__()
        # 状态编码
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        
        # 动作编码
        self.fc2 = torch.nn.Linear(action_dim, hidden_dim)
        self.ln2 = torch.nn.LayerNorm(hidden_dim)
        
        # 合并层
        self.fc3 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln3 = torch.nn.LayerNorm(hidden_dim)
        
        # 输出层
        self.fc4 = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(p=0.1)
        
    def forward(self, state, action):
        # 处理状态
        s = F.relu(self.fc1(state))
        s = self.ln1(s)
        s = self.dropout(s)
        
        # 处理动作
        a = F.relu(self.fc2(action))
        a = self.ln2(a)
        a = self.dropout(a)
        
        # 合并状态和动作
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc3(x))
        x = self.ln3(x)
        x = self.dropout(x)
        
        # 输出Q值
        return self.fc4(x) 