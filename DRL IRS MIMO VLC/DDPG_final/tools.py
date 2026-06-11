import numpy as np
import os
import torch
import copy

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)

    def store_transition(self, state, action, reward, new_state):
        """存储一条经验"""
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """随机采样一批经验"""
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]

        return states, actions, rewards, states_
    
    def __len__(self):
        """返回当前缓冲区大小"""
        return min(self.mem_cntr, self.mem_size) 

def save_models(actor, critic, target_actor, target_critic, path):
    """保存模型参数到指定路径
    
    参数:
    actor -- Actor网络
    critic -- Critic网络
    target_actor -- 目标Actor网络
    target_critic -- 目标Critic网络
    path -- 保存路径
    """
    # 确保目录存在并替换路径分隔符，解决Windows路径问题
    path = path.replace('\\', '/')
    os.makedirs(path, exist_ok=True)
    
    try:
        # 保存模型参数
        actor_path = os.path.join(path, "actor.pth").replace('\\', '/')
        critic_path = os.path.join(path, "critic.pth").replace('\\', '/')
        target_actor_path = os.path.join(path, "target_actor.pth").replace('\\', '/')
        target_critic_path = os.path.join(path, "target_critic.pth").replace('\\', '/')
        
        torch.save(actor.state_dict(), actor_path)
        torch.save(critic.state_dict(), critic_path)
        torch.save(target_actor.state_dict(), target_actor_path)
        torch.save(target_critic.state_dict(), target_critic_path)
        print(f"模型已保存到 {path}")
    except Exception as e:
        print(f"保存模型时出错: {e}")
        print(f"尝试的保存路径: {path}")

def load_models(actor, critic, target_actor, target_critic, path):
    """从指定路径加载模型参数
    
    参数:
    actor -- Actor网络
    critic -- Critic网络
    target_actor -- 目标Actor网络
    target_critic -- 目标Critic网络
    path -- 加载路径
    
    返回:
    bool -- 是否成功加载
    """
    # 替换路径分隔符，解决Windows路径问题
    path = path.replace('\\', '/')
    
    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"警告: 路径 {path} 不存在，无法加载模型")
        return False
        
    try:
        # 加载模型参数
        actor_path = os.path.join(path, "actor.pth").replace('\\', '/')
        critic_path = os.path.join(path, "critic.pth").replace('\\', '/')
        target_actor_path = os.path.join(path, "target_actor.pth").replace('\\', '/')
        target_critic_path = os.path.join(path, "target_critic.pth").replace('\\', '/')
        
        actor.load_state_dict(torch.load(actor_path))
        critic.load_state_dict(torch.load(critic_path))
        target_actor.load_state_dict(torch.load(target_actor_path))
        target_critic.load_state_dict(torch.load(target_critic_path))
        print(f"模型已从 {path} 加载")
        return True
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print(f"尝试的加载路径: {path}")
        return False 

def sort_action(action, num):

    data = copy.deepcopy(action)
    index = []
    if type(data) is np.ndarray:
        max_num = np.max(data)
    else:
        max_num = torch.max(data)
    for n in range(num):
        index.append(np.argmax(data))
        data[np.argmax(data)] = -1e10

    return index, max_num


class EnhancedReplayBuffer(ReplayBuffer):
    """增强版经验回放缓冲区，支持低质量样本移除功能"""

    def __init__(self, max_size, input_shape, n_actions):
        super().__init__(max_size, input_shape, n_actions)
        # 初始化一个数组用于存储所有奖励值
        self.reward_storage = np.zeros(max_size, dtype=np.float32)
        # 记录有效样本的索引
        self.valid_indices = np.arange(max_size)
        self.valid_count = 0

    def store_transition(self, state, action, reward, new_state):
        """存储一条经验"""
        # 使用父类方法存储
        super().store_transition(state, action, reward, new_state)

        # 获取当前存储位置（考虑循环缓冲区）
        index = (self.mem_cntr - 1) % self.mem_size

        # 存储奖励值
        self.reward_storage[index] = reward

        # 更新有效索引
        if self.valid_count < self.mem_size:
            self.valid_count += 1

    def sample_buffer(self, batch_size):
        """随机采样一批经验（只从有效样本中采样）"""
        # 确保不超过有效样本数
        batch_size = min(batch_size, self.valid_count)
        batch = np.random.choice(self.valid_indices[:self.valid_count], batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]

        return states, actions, rewards, states_

    def remove_low_reward_samples(self, reward_threshold):
        """移除低于指定阈值的样本

        参数:
        reward_threshold -- 奖励阈值，低于此值的样本将被移除
        """
        if self.valid_count == 0:
            print("缓冲区为空，无法移除样本")
            return 0

        # 找出低奖励样本的索引
        low_reward_mask = self.reward_storage[:self.valid_count] < reward_threshold
        low_reward_indices = np.where(low_reward_mask)[0]

        # 如果没有低奖励样本
        if len(low_reward_indices) == 0:
            print(f"没有低于阈值 {reward_threshold} 的样本")
            return 0

        # 找出高奖励样本的索引
        high_reward_indices = np.where(~low_reward_mask)[0]

        # 创建新的有效索引数组
        new_valid_indices = np.empty_like(self.valid_indices)
        new_valid_count = len(high_reward_indices)

        # 将高奖励样本移动到数组前面
        new_valid_indices[:new_valid_count] = high_reward_indices

        # 更新有效样本计数
        self.valid_count = new_valid_count

        # 打印移除信息
        removed_count = len(low_reward_indices)
        print(f"--移除了 {removed_count} 个低奖励样本（阈值: {reward_threshold:.2f}），剩余 {self.valid_count} 个样本--\n")

        return removed_count

    def get_reward_statistics(self):
        """获取奖励统计信息"""
        if self.valid_count == 0:
            return 0, 0, 0

        rewards = self.reward_storage[:self.valid_count]
        return np.min(rewards), np.max(rewards), np.mean(rewards)

    def __len__(self):
        """返回当前有效缓冲区大小"""
        return self.valid_count


def save_models(actor, critic, target_actor, target_critic, path):
    """保存模型参数到指定路径"""
    # 确保目录存在并替换路径分隔符，解决Windows路径问题
    path = path.replace('\\', '/')
    os.makedirs(path, exist_ok=True)

    try:
        # 保存模型参数
        actor_path = os.path.join(path, "actor.pth").replace('\\', '/')
        critic_path = os.path.join(path, "critic.pth").replace('\\', '/')
        target_actor_path = os.path.join(path, "target_actor.pth").replace('\\', '/')
        target_critic_path = os.path.join(path, "target_critic.pth").replace('\\', '/')

        torch.save(actor.state_dict(), actor_path)
        torch.save(critic.state_dict(), critic_path)
        torch.save(target_actor.state_dict(), target_actor_path)
        torch.save(target_critic.state_dict(), target_critic_path)
        print(f"模型已保存到 {path}")
    except Exception as e:
        print(f"保存模型时出错: {e}")
        print(f"尝试的保存路径: {path}")


def load_models(actor, critic, target_actor, target_critic, path):
    """从指定路径加载模型参数"""
    # 替换路径分隔符，解决Windows路径问题
    path = path.replace('\\', '/')

    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"警告: 路径 {path} 不存在，无法加载模型")
        return False

    try:
        # 加载模型参数
        actor_path = os.path.join(path, "actor.pth").replace('\\', '/')
        critic_path = os.path.join(path, "critic.pth").replace('\\', '/')
        target_actor_path = os.path.join(path, "target_actor.pth").replace('\\', '/')
        target_critic_path = os.path.join(path, "target_critic.pth").replace('\\', '/')

        actor.load_state_dict(torch.load(actor_path))
        critic.load_state_dict(torch.load(critic_path))
        target_actor.load_state_dict(torch.load(target_actor_path))
        target_critic.load_state_dict(torch.load(target_critic_path))
        print(f"模型已从 {path} 加载")
        return True
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print(f"尝试的加载路径: {path}")
        return False


def sort_action(action, num):
    data = copy.deepcopy(action)
    index = []
    if type(data) is np.ndarray:
        max_num = np.max(data)
    else:
        max_num = torch.max(data)
    for n in range(num):
        index.append(np.argmax(data))
        data[np.argmax(data)] = -1e10

    return index, max_num