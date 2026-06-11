import torch
import torch.nn.functional as F
from DNN import PolicyNet, ValueNet
from tools import EnhancedReplayBuffer  # 使用增强版回放缓冲区
import numpy as np
import os
import copy

class Agent_DDPG:
    def __init__(self, hidden_dim, sigma, actor_lr, critic_lr, tau, gamma, device, env, batch_size,
                 buffer_size , sigma_min, decay_rate):
        self.env = env
        self.action_dim = env.action_dim
        self.state_dim = env.state_dim
        
        # 初始化策略网络和价值网络
        self.actor = PolicyNet(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.critic = ValueNet(self.state_dim, hidden_dim, self.action_dim).to(device)
        self.target_actor = PolicyNet(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_critic = ValueNet(self.state_dim, hidden_dim, self.action_dim).to(device)
        
        # 加载目标网络参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 保存初始学习率
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        # 超参数
        self.gamma = gamma  # 折扣因子
        self.sigma = sigma  # 探索噪声初始值
        self.sigma_min = sigma_min  # 噪声最小值
        self.decay_rate = decay_rate  # 噪声衰减率
        self.tau = tau      # 软更新系数
        self.device = device
        self.batch_size = batch_size  # 批处理大小
        self.buffer_size = buffer_size

        # 经验回放缓冲区 - 使用增强版
        self.memory = EnhancedReplayBuffer(self.buffer_size, self.state_dim, self.action_dim)
        self.mean_loss = 0

    def take_action(self, state, explore=True):
        """根据当前状态选择动作"""
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]  # 改成一维数组
        
        # 添加探索噪声
        if explore:
            noise = np.random.normal(0, self.sigma, size=self.action_dim)
            action += noise
        
        # 返回处理后的动作
        return action

    def decay_noise(self):
        """衰减探索噪声"""
        self.sigma = max(self.sigma_min, self.sigma * self.decay_rate)
        return self.sigma
    
    #学习率衰减
    def decay_learning_rate(self, decay_rate, min_actor_lr, min_critic_lr):
        """衰减学习率
        
        参数:
        decay_rate -- 学习率衰减率
        min_actor_lr -- Actor网络的最小学习率
        min_critic_lr -- Critic网络的最小学习率
        """
        # 获取当前学习率
        for param_group in self.actor_optimizer.param_groups:
            new_lr = max(min_actor_lr, param_group['lr'] * decay_rate)
            param_group['lr'] = new_lr
            self.actor_lr = new_lr
            
        for param_group in self.critic_optimizer.param_groups:
            new_lr = max(min_critic_lr, param_group['lr'] * decay_rate)
            param_group['lr'] = new_lr
            self.critic_lr = new_lr
            
        return self.actor_lr, self.critic_lr
    
    def get_learning_rates(self):
        """获取当前学习率"""
        return self.actor_lr, self.critic_lr
    
    def update(self):
        """更新网络参数"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # 从经验回放中采样
        states, actions, rewards, next_states = self.memory.sample_buffer(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # 计算目标Q值
        next_actions = self.target_actor(next_states)
        target_q = self.target_critic(next_states, next_actions)
        target_q = rewards + self.gamma * target_q
        
        # 计算当前Q值
        current_q = self.critic(states, actions)
        
        # 计算critic损失
        critic_loss = F.mse_loss(current_q, target_q)
        
        # 更新critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 计算actor损失
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # 更新actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        
        return critic_loss.item()

    def soft_update(self, net, target_net):
        """软更新目标网络参数"""
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def remember(self, state, action, reward, next_state):
        """存储经验到回放缓冲区"""
        self.memory.store_transition(state, action, reward, next_state)


class StateRollback:
    def __init__(self,
                 agent,
                 checkpoint_freq,
                 collapse_threshold,
                 window_size,
                 threshold_ratio,
                 checkpoint_dir='ddpg_checkpoints',):
        """
        初始化状态回滚系统
        """
        self.agent = agent
        self.device = agent.device

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.collapse_threshold = collapse_threshold
        self.window_size = window_size
        self.threshold_ratio = threshold_ratio

        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 初始化监控变量
        self.reward_history = []
        self.best_avg_reward = -np.inf
        self.last_saved_episode = 0

    def save_checkpoint(self, episode, reward):
        """保存当前状态为检查点"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_ep{episode}_r{reward:.2f}.pt'
        )

        # 保存缓冲区状态
        buffer_state = {
            'state_memory': self.agent.memory.state_memory.copy(),
            'new_state_memory': self.agent.memory.new_state_memory.copy(),
            'action_memory': self.agent.memory.action_memory.copy(),
            'reward_memory': self.agent.memory.reward_memory.copy(),
            'reward_storage': self.agent.memory.reward_storage.copy(),
            'valid_indices': self.agent.memory.valid_indices.copy(),
            'mem_cntr': self.agent.memory.mem_cntr,
            'valid_count': self.agent.memory.valid_count
        }

        torch.save({
            'episode': episode,
            'reward': reward,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'target_actor_state_dict': self.agent.target_actor.state_dict(),
            'target_critic_state_dict': self.agent.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
            'sigma': self.agent.sigma,
            'reward_history': copy.deepcopy(self.reward_history),
            'buffer_state': buffer_state
        }, checkpoint_path)

        self.last_saved_episode = episode
        print(f"检查点已保存到: {checkpoint_path}\n")
        return checkpoint_path

    def load_latest_checkpoint(self):
        """加载最新的稳定检查点"""
        # 获取所有检查点并按时间排序
        checkpoints = [f for f in os.listdir(self.checkpoint_dir)
                       if f.startswith('checkpoint') and f.endswith('.pt')]

        if not checkpoints:
            raise ValueError("没有可用的检查点用于回滚")

        # 按奖励值排序（取性能最好的）
        checkpoints.sort(key=lambda x: float(x.split('_r')[1].split('.pt')[0]), reverse=True)
        best_checkpoint = os.path.join(self.checkpoint_dir, checkpoints[0])

        print(f"从检查点恢复: {best_checkpoint}\n")
        checkpoint = torch.load(best_checkpoint, map_location=self.device, weights_only=False)

        # 恢复网络参数
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.agent.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])

        # 恢复优化器状态
        self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # 恢复其他训练状态
        self.agent.sigma = checkpoint['sigma']

        # 恢复缓冲区状态
        buffer_state = checkpoint['buffer_state']
        self.agent.memory.state_memory = buffer_state['state_memory']
        self.agent.memory.new_state_memory = buffer_state['new_state_memory']
        self.agent.memory.action_memory = buffer_state['action_memory']
        self.agent.memory.reward_memory = buffer_state['reward_memory']
        self.agent.memory.reward_storage = buffer_state['reward_storage']
        self.agent.memory.valid_indices = buffer_state['valid_indices']
        self.agent.memory.mem_cntr = buffer_state['mem_cntr']
        self.agent.memory.valid_count = buffer_state['valid_count']

        # 恢复奖励历史
        self.reward_history = checkpoint['reward_history']
        self.best_avg_reward = np.mean(self.reward_history[-self.window_size:])

        return checkpoint['episode'], checkpoint['reward']

    def monitor_performance(self, episode, reward):
        """监控性能并决定是否需要回滚"""
        self.reward_history.append(reward)

        # 计算滑动平均奖励
        if len(self.reward_history) >= self.window_size:
            recent_rewards = self.reward_history[-self.window_size:]
            avg_reward = np.mean(recent_rewards)

            # 更新最佳平均奖励
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward

            # 检测崩溃：当前奖励显著低于历史最佳
            if reward < self.collapse_threshold * self.best_avg_reward:
                print(f"---检测到崩溃在回合 {episode+1}! 正在回滚...---\n")
                return True

        # 定期保存检查点
        if episode - self.last_saved_episode >= self.checkpoint_freq:
            self.save_checkpoint(episode, reward)

        return False


    def post_rollback_adjustment(self):
         """回滚后的超参数调整"""
         # 1. 降低学习率
         for param_group in self.agent.actor_optimizer.param_groups:
             param_group['lr'] *= 0.9992
         for param_group in self.agent.critic_optimizer.param_groups:
             param_group['lr'] *= 0.9992
         print(f"--学习率已降低: Actor={self.agent.actor_optimizer.param_groups[0]['lr']:.6f}, "
               f"Critic={self.agent.critic_optimizer.param_groups[0]['lr']:.6f}--\n")

         # 2. 减小探索噪声

         # 3. 清空回放池中的低质量数据
         min_r, max_r, avg_r = self.agent.memory.get_reward_statistics()
         threshold = avg_r * self.threshold_ratio
         removed = self.agent.memory.remove_low_reward_samples(threshold)

         # 4. 重置总训练步数以匹配回滚点
         return removed
