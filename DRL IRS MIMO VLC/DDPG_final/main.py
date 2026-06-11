import numpy as np
#import torch
import argparse
import os
#import sys
import matplotlib.pyplot as plt
from RIS_env import RIS_env, Room_env
from DDPG import Agent_DDPG,StateRollback
from device_setup import setup_device, print_gpu_memory_status
import datetime  # 导入datetime模块用于生成时间戳
import time  # 用于计时

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DDPG算法训练IRS辅助MIMO VLC系统')
    parser.add_argument('--episodes', type=int, default=500, help='训练回合数')
    parser.add_argument('--max_steps', type=int, default=100, help='每个回合的最大步数')
    parser.add_argument('--hidden_dim', type=int, default=300, help='神经网络隐藏层维度')

    parser.add_argument('--sigma', type=float, default=0.0003, help='探索噪声参数')
    parser.add_argument('--sigma_min', type=float, default=0.00003, help='最小探索噪声')
    parser.add_argument('--sigma_max', type=float, default=0.003, help='最大探索噪声')
    parser.add_argument('--decay_rate', type=float, default=0.996, help='噪声衰减率')
    parser.add_argument('--up_rate', type=float, default=1.006, help='奖励小于过去平均奖励零时 噪声提升率')
    parser.add_argument('--down_rate', type=float, default=0.996, help='奖励大于过去平均奖励零时 噪声减少率')

    parser.add_argument('--gamma', type=float, default=0.985, help='折扣因子')
    parser.add_argument('--actor_lr', type=float, default=1e-5, help='Actor学习率')
    parser.add_argument('--critic_lr', type=float, default=3e-4, help='Critic学习率')

    parser.add_argument('--tau', type=float, default=1e-4, help='软更新参数')
    parser.add_argument('--save_dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--buffer_size', type=int, default=120000, help='经验回放缓冲区大小')
    parser.add_argument('--lr_decay', type=float, default=0.994, help='学习率衰减率')
    parser.add_argument('--lr_decay_freq', type=int, default=20, help='学习率衰减频率(每多少个回合)')

    parser.add_argument('--min_actor_lr', type=float, default=1e-6, help='Actor最小学习率')
    parser.add_argument('--min_critic_lr', type=float, default=1e-5, help='Critic最小学习率')
    # 添加回滚相关参数
    parser.add_argument('--threshold_ratio', type=float, default=0.98, help='reply buffer删除的阈值d的比例')
    parser.add_argument('--checkpoint_freq', type=int, default=3, help='检查点保存频率(每多少个回合)')
    parser.add_argument('--collapse_threshold', type=float, default=0.98, help='崩溃检测阈值(当前奖励/历史平均)')
    parser.add_argument('--window_size', type=int, default=5, help='滑动平均窗口大小')
    return parser.parse_args()

# 参数描述字典，用于保存配置文件
PARAM_DESCRIPTIONS = {
    "episodes": "训练回合数",
    "max_steps": "每个回合的最大步数",
    "hidden_dim": "神经网络隐藏层维度",
    "sigma": "探索噪声参数",
    "sigma_min": "最-小-探索噪声",
    'sigma_max': '最-大-探索噪声',
    "decay_rate": "噪声衰减率",
    "up_rate":"奖励差-大-于零时噪声提升率",
    "down_rate":"奖励差-小-于零时噪声提升率",
    "gamma": "折扣因子",
    "actor_lr": "Actor学习率",
    "critic_lr": "Critic学习率",
    "tau": "软更新参数",
    "batch_size": "批处理大小",
    "buffer_size": "经验回放缓冲区大小",
    "lr_decay": "学习率衰减率",
    "lr_decay_freq": "学习率衰减频率",
    "min_actor_lr": "Actor最小学习率",
    "min_critic_lr": "Critic最小学习率",
    "checkpoint_freq": "检查点保存频率",
    'threshold_ratio': 'reply buffer删除的阈值',
    "collapse_threshold": "崩溃检测阈值",
    "window_size": "滑动平均窗口大小"
}

def save_config(args, save_dir):
    """将训练参数保存到配置文件中
    
    参数:
    args -- 命令行参数对象
    save_dir -- 保存目录
    """
    config_file = os.path.join(save_dir, "DDPG-Configuration.txt")
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("\n")
        for arg, value in vars(args).items():
            description = PARAM_DESCRIPTIONS.get(arg, arg)
            f.write(f"--{arg} = {value}, '{description}'\n")
    
    print(f"配置参数已保存到: {config_file}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建带有时间戳的保存目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果将保存到: {save_dir}")
    
    # 设定参数
    M = 4       # 灯的个数
    N = 4      # 用户数
    K = 25      # IRS元件个数
    mLED = 2
    nPD = 2
    kIRS = 5
    
    # 使用命令行解析的超参数
    hidden_dim = args.hidden_dim
    sigma = args.sigma
    sigma_min = args.sigma_min
    sigma_max = args.sigma_max
    decay_rate = args.decay_rate
    up_rate = args.up_rate
    down_rate = args.down_rate
    gamma = args.gamma
    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    tau = args.tau
    episodes = args.episodes
    max_steps = args.max_steps
    batch_size = args.batch_size
    buffer_size = args.buffer_size
    lr_decay = args.lr_decay
    lr_decay_freq = args.lr_decay_freq
    min_actor_lr = args.min_actor_lr
    min_critic_lr = args.min_critic_lr
    checkpoint_freq = args.checkpoint_freq
    threshold_ratio = args.threshold_ratio
    collapse_threshold = args.collapse_threshold
    window_size = args.window_size

    # 设置训练设备并显示GPU状态
    device = setup_device()
    
    # 初始化数据存储列表
    ep_reward_list = []
    ep_loss_list = []
    avg_score = np.zeros(episodes)
    avg_loss = np.zeros(episodes)
    
    # 设定环境和智能体
    env_Room = Room_env(M, N, K, mLED, nPD, kIRS)
    env = RIS_env(env_Room)
    agent = Agent_DDPG(hidden_dim, sigma, actor_lr, critic_lr, tau, gamma, device, env, 
                       batch_size=batch_size, buffer_size=buffer_size, sigma_min=sigma_min, 
                       decay_rate=decay_rate)
    
    print("开始训练DDPG算法...")
    print(f"设备: {device}")
    print(f"参数设置: M={M}, N={N}, K={K}, 所有灯都激活")
    print(f"训练回合数: {episodes}, 每回合步数: {max_steps}")
    print(f"批处理大小: {batch_size}, 缓冲区大小: {buffer_size}")
    print(f"学习率衰减: 初始Actor学习率={actor_lr}, Critic学习率={critic_lr}, 衰减率={lr_decay}, 频率={lr_decay_freq}回合")

    # 初始化状态回滚管理器（如果启用）
    rollback_manager = None
    enable_rollback = True
    if enable_rollback:
        rollback_manager = StateRollback(
            agent,
            checkpoint_dir=os.path.join(save_dir, 'checkpoints'),
            checkpoint_freq=checkpoint_freq,
            collapse_threshold=collapse_threshold,
            window_size=window_size,
            threshold_ratio=threshold_ratio
        )
        # 保存初始检查点
        rollback_manager.save_checkpoint(0, 0)
        print(f"--状态回滚已启用: 检查点频率={checkpoint_freq}, 崩溃阈值={collapse_threshold}, 窗口大小={window_size}--\n")

    # 保存训练配置参数
    save_config(args, save_dir)

    start_time = time.time()

    # 开始训练循环
    for i in range(episodes):
        # 环境初始化
        if (i + 1) % 2 == 0:
            env.threshold = env.threshold - 0.0015
        state, _ = env.reset()
        episode_reward = 0
        episode_start_time = time.time()
        for step in range(max_steps):

            # 智能体选择动作
            action = agent.take_action(state, explore=True)
            
            # 环境执行动作
            reward, next_state, done = env.step(action)

            #if reward > 0:
            #    print(f"[回合 {i+1}, 步数 {step+1}] 找到有效动作！奖励值: {reward:.4f}")
            
            # 存储经验
            agent.remember(state, action, reward, next_state)
            
            # 更新网络
            loss = agent.update()
            if loss > 0:
                ep_loss_list.append(loss)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            
            # 记录奖励
            ep_reward_list.append(reward)
            # 如果找到有效解决方案，可以提前结束本回合
            #if done:
            #    print(f"回合 {i+1} 在步数 {step+1} 找到有效解决方案，提前结束！")
            #    break

        # 在每个回合结束后衰减噪声
        #current_sigma = agent.decay_noise()

        # 计算平均奖励和损失
        avg_score[i] = np.mean(ep_reward_list[-100:]) if len(ep_reward_list) > 0 else 0

        # 基于近期奖励变化率调整噪声
        reward_diff = avg_score[i]-avg_score[i-1]
        print('reward_diff',reward_diff)
        if reward_diff > 0:
            agent.sigma = max(agent.sigma * down_rate, sigma_min)  # 减小噪声
            print('减少噪声')
        else:
            agent.sigma = min(agent.sigma * up_rate, sigma_max)  # 增大噪声
            print('增大噪声')

        if len(ep_loss_list) > 0:
            avg_loss[i] = np.mean(ep_loss_list[-100:])
        else:
            avg_loss[i] = 0

        # 学习率衰减（每lr_decay_freq个回合）
        if (i+1) % lr_decay_freq == 0:
            agent.decay_learning_rate(lr_decay, min_actor_lr, min_critic_lr)
            current_actor_lr, current_critic_lr = agent.get_learning_rates()

        # 计算回合时间
        episode_time = time.time() - episode_start_time
        print(f'回合 {i + 1}/{episodes}, 耗时: {episode_time:.2f}秒, 损失: {avg_loss[i]:.4f}, '
                  f'平均奖励: {avg_score[i]:.4f}, 本回合奖励: {episode_reward:.4f}, '
                  f'噪声: {agent.sigma:.4f}')

        # 每  ?个回合保存一次训练曲线
        if (i + 1) % 100 == 0:
            # 保存并绘制训练曲线
            plot_file = os.path.join(save_dir, f'平均奖励,第{i + 1}回合.png')
            plot_training_curve(range(i + 1), avg_score[:i + 1], plot_file)

        # 监控性能并处理回滚（如果启用）
        if rollback_manager is not None:
            if rollback_manager.monitor_performance(i, episode_reward):
                try:
                    rollback_episode, rollback_reward = rollback_manager.load_latest_checkpoint()
                    removed_samples = rollback_manager.post_rollback_adjustment()
                    print(f" ^3^ 已回滚到回合 {rollback_episode} —— 奖励: {rollback_reward:.2f} ——, "
                          f"移除了 {removed_samples} 个低质量样本\n")

                    # 跳过崩溃后的部分训练
                    i = rollback_episode - 1  # -1因为循环会+1
                    continue  # 跳过本轮剩余代码，直接开始下一轮
                except Exception as e:
                    print(f"回滚失败: {e}")


    # 训练结束后绘制最终训练曲线
    final_plot_file = os.path.join(save_dir, '最终平均奖励.png')
    plot_training_curve(range(episodes), avg_score, final_plot_file)

    # 保存 avg_score 和 avg_loss 数据
    np.save(os.path.join(save_dir, 'avg_score.npy'), avg_score)
    np.save(os.path.join(save_dir, 'avg_loss.npy'), avg_loss)
    
    # 保存训练好的模型
    #model_save_path = os.path.join(save_dir, 'models')
    #agent.save_models(model_save_path)
    # 训练结束后计算总时间
    total_time = time.time() - start_time
    print(f"---训练完成! 总耗时: {total_time / 60:.2f}分钟---")

    # 如果启用了回滚，保存最后一个检查点
    if rollback_manager is not None:
        rollback_manager.save_checkpoint(episodes, avg_score[-1])

def plot_training_curve(x, y, filename):
    """绘制并保存训练曲线图"""
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Average Score')

    # 添加移动平均线
    window_size = 10
    moving_avg = np.convolve(y, np.ones(window_size) / window_size, mode='valid')
    plt.plot(x[window_size - 1:], moving_avg, 'r-', label=f'Moving Avg ({window_size} eps)')

    plt.title('Training Performance Curve (DDPG)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"训练曲线图已保存到: {filename}")

if __name__ == "__main__":
    main()