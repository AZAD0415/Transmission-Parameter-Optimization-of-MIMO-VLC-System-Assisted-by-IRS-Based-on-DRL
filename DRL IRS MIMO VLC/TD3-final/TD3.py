import torch
import torch.nn.functional as F
import torch.serialization
from DNN import PolicyNet, ValueNet
import numpy as np
import os
from tools import EnhancedReplayBuffer
import copy
class Agent_TD3:
    def __init__(self, hidden_dim, sigma, actor_lr, critic_lr, tau, gamma, device, env, batch_size,
                 buffer_size, sigma_min , decay_rate,
                 policy_noise, noise_clip, policy_freq ):
        
        self.env = env
        self.action_dim = env.action_dim
        self.state_dim = env.state_dim
        # TD3需要知道动作范围以便进行裁剪
        self.max_action = env.max_action
        
        # 初始化策略网络和价值网络 (TD3: 1个actor, 2个critic)
        self.actor = PolicyNet(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.critic1 = ValueNet(self.state_dim, hidden_dim, self.action_dim).to(device)
        self.critic2 = ValueNet(self.state_dim, hidden_dim, self.action_dim).to(device)
        
        self.target_actor = PolicyNet(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_critic1 = ValueNet(self.state_dim, hidden_dim, self.action_dim).to(device)
        self.target_critic2 = ValueNet(self.state_dim, hidden_dim, self.action_dim).to(device)
        
        # 加载目标网络参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # TD3: 两个critic网络使用一个优化器
        self.critic_optimizer = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)
        
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

        # TD3 specific hyperparameters
        self.policy_noise = policy_noise   # 策略噪声
        self.noise_clip = noise_clip       # 噪声裁剪范围
        self.policy_freq = policy_freq     # 策略网络延迟更新频率
        self.total_it = 0                  # 总训练步数

        self.memory = EnhancedReplayBuffer(buffer_size, self.state_dim, self.action_dim)

    def take_action(self, state, explore=True):
        """根据当前状态选择动作"""
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        
        # 添加探索噪声
        if explore:
            noise = np.random.normal(0, self.sigma * self.max_action, size=self.action_dim)
            action += noise
        
        # 裁剪动作到有效范围内
        return action.clip(-self.max_action, self.max_action)

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
            
        self.total_it += 1
        
        # 从经验回放中采样
        states, actions, rewards, next_states = self.memory.sample_buffer(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # --- TD3核心：目标策略平滑 ---
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            # 计算带噪声的目标动作并裁剪
            next_actions = (self.target_actor(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            # --- TD3核心：裁剪双Q学习 ---
            # 计算两个目标critic网络的目标Q值
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            
            # 取较小的Q值作为最终目标
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * target_q
        
        # 计算当前Q值
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # 计算critic损失 (两个critic的损失之和)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- TD3核心：延迟策略更新 ---
        if self.total_it % self.policy_freq == 0:
            
            # 计算actor损失 (使用critic1)
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            # 更新actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic1, self.target_critic1)
            self.soft_update(self.critic2, self.target_critic2)
        
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
                 agent,  # 传入TD3智能体实例
                 checkpoint_freq,
                 collapse_threshold,
                 window_size,
                 threshold_ratio,
                 checkpoint_dir='td3_checkpoints'):

        #checkpoint_freq=100
        #collapse_threshold=0.7
        #window_size=20


        """
        初始化状态回滚系统

        参数:
        - agent: TD3智能体实例
        - checkpoint_dir: 检查点保存目录
        - checkpoint_freq: 保存检查点的回合间隔
        - collapse_threshold: 崩溃阈值 (当前奖励/历史平均)
        - window_size: 滑动平均窗口大小
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
            'critic1_state_dict': self.agent.critic1.state_dict(),
            'critic2_state_dict': self.agent.critic2.state_dict(),
            'target_actor_state_dict': self.agent.target_actor.state_dict(),
            'target_critic1_state_dict': self.agent.target_critic1.state_dict(),
            'target_critic2_state_dict': self.agent.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
            'total_it': self.agent.total_it,
            #'sigma': self.agent.sigma,
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
        self.agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.agent.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.agent.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.agent.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])

        # 恢复优化器状态
        self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # 恢复其他训练状态
        self.agent.total_it = checkpoint['total_it']
        #self.agent.sigma = checkpoint['sigma']

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
            recent_rewards = self.reward_history[-self.window_size:] #计算储存的最近checkpoint_freq个样本中window_size个点
            avg_reward = np.mean(recent_rewards)

            # 更新最佳平均奖励
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward

            # 检测崩溃：当前奖励显著低于历史最佳
            if reward < self.collapse_threshold * self.best_avg_reward: # 判断是否小于回滚阈值
                print(f"！！！检测到回合 -{episode+1}-崩溃， 正在回滚...！！！\n")
                return True

        # 定期保存检查点
        if episode - self.last_saved_episode >= self.checkpoint_freq:
            self.save_checkpoint(episode, reward)

        return False

    def post_rollback_adjustment(self):
        """回滚后的超参数调整"""
        # 1. 降低学习率
        for param_group in self.agent.actor_optimizer.param_groups:
            param_group['lr'] *= 0.999
        for param_group in self.agent.critic_optimizer.param_groups:
            param_group['lr'] *= 0.999
        print(f"--学习率已降低: Actor={self.agent.actor_optimizer.param_groups[0]['lr']:.6f}, "
              f"Critic={self.agent.critic_optimizer.param_groups[0]['lr']:.6f}--\n")

        # 2. 减小探索噪声
        #self.agent.sigma = max(self.agent.sigma_min, self.agent.sigma * 0.985)
        #print(f"--探索噪声已降低: {self.agent.sigma:.4f}--\n")

        # 3. 清空回放池中的低质量数据
        min_r, max_r, avg_r = self.agent.memory.get_reward_statistics()
        threshold = avg_r * self.threshold_ratio
        removed = self.agent.memory.remove_low_reward_samples(threshold)

        # 4. 增加目标网络更新延迟（降低策略更新频率）
        original_freq = self.agent.policy_freq
        self.agent.policy_freq = min(2, self.agent.policy_freq + 1)
        print(f"--策略更新频率从 {original_freq} 增加到 {self.agent.policy_freq}--\n")

        # 5. 重置总训练步数以匹配回滚点
        self.agent.total_it = 0

        return removed