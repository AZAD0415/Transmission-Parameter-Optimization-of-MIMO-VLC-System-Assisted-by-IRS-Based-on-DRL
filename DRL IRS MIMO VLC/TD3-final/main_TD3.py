
import numpy as np
import torch
from TD3 import Agent_TD3, StateRollback # 导入StateRollback
import matplotlib.pyplot as plt
import os
import argparse
import sys
from RIS_env import RIS_env, Room_env
from device_setup import setup_device, print_gpu_memory_status
import datetime
import time  # 用于计时

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TD3算法训练IRS辅助MIMO VLC系统')
    parser.add_argument('--episodes', type=int, default=500, help='训练回合数')
    parser.add_argument('--max_steps', type=int, default=100, help='每个回合的最大步数')
    parser.add_argument('--hidden_dim', type=int, default=300, help='神经网络隐藏层维度')

    parser.add_argument('--sigma', type=float, default=0.0003, help='探索噪声参数')
    parser.add_argument('--sigma_min', type=float, default=0.00003, help='最小探索噪声')
    parser.add_argument('--sigma_max', type=float, default=0.003, help='最大探索噪声')
    parser.add_argument('--decay_rate', type=float, default=0.99, help='噪声衰减率')

    parser.add_argument('--up_rate', type=float, default=1.01, help='奖励小于过去平均奖励零时 噪声提升率')
    parser.add_argument('--down_rate', type=float, default=0.99, help='奖励大于过去平均奖励零时 噪声减少率')

    parser.add_argument('--gamma', type=float, default=0.985, help='折扣因子')
    parser.add_argument('--actor_lr', type=float, default=1e-5, help='Actor学习率')
    parser.add_argument('--critic_lr', type=float, default=3e-4, help='Critic学习率')

    parser.add_argument('--tau', type=float, default=1e-4, help='软更新参数')
    parser.add_argument('--save_dir', type=str, default='results_td3', help='结果保存目录')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--buffer_size', type=int, default=120000, help='经验回放缓冲区大小')

    parser.add_argument('--lr_decay', type=float, default=0.985, help='学习率衰减率')
    parser.add_argument('--lr_decay_freq', type=int, default=50, help='学习率衰减频率(每多少个回合)')

    parser.add_argument('--min_actor_lr', type=float, default=1e-6, help='Actor最小学习率')
    parser.add_argument('--min_critic_lr', type=float, default=1e-6, help='Critic最小学习率')

    parser.add_argument('--policy_noise', type=float, default=0.1, help='TD3 策略 噪声')
    parser.add_argument('--noise_clip', type=float, default=0.001, help='TD3 噪声裁剪')
    parser.add_argument('--max_action', type=float, default=1.5, help='最大动作值' )
    parser.add_argument('--policy_freq', type=int, default=3, help='TD3策略延迟更新频率')

    # 添加状态回滚相关参数
    parser.add_argument('--threshold_ratio',type=float, default=0.99, help='reply buffer删除的阈值d的比例' )
    parser.add_argument('--checkpoint_freq', type=int, default=4, help='检查点保存频率(每多少个回合)')
    parser.add_argument('--collapse_threshold', type=float, default=0.991, help='崩溃检测阈值(当前奖励/历史平均)')
    parser.add_argument('--window_size', type=int, default=5, help='滑动平均窗口大小')
    #parser.add_argument('--enable_rollback', action='store_true', help='启用状态回滚功能')
    # python main_TD3.py --enable_rollback
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
    "policy_noise": "TD3策略噪声",
    "noise_clip": "TD3噪声裁剪",
    "policy_freq": "TD3策略延迟更新频率",
    "checkpoint_freq": "检查点保存频率",
    'threshold_ratio': 'reply buffer删除的阈值',
    "collapse_threshold": "崩溃检测阈值",
    "window_size": "滑动平均窗口大小",
}

def save_config(args, save_dir):
    """将训练参数保存到配置文件中
    
    参数:
    args -- 命令行参数对象
    save_dir -- 保存目录
    """
    config_file = os.path.join(save_dir, "TD3-Configration.txt")
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
    N = 4       # 用户数
    K = 36      # RIS元件个数
    mLED = 2
    nPD = 2
    kIRS= 3

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
    policy_noise = args.policy_noise
    noise_clip = args.noise_clip
    policy_freq = args.policy_freq
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
    env_Room = Room_env(M, N, K, mLED , nPD, kIRS)
    env = RIS_env(env_Room)

    # c超参数设置动作最大值
    env.max_action = args.max_action

    # --- 修改: 实例化TD3智能体 ---
    # 移除 state_dim 和 action_dim 参数，Agent内部会从env获取
    agent = Agent_TD3(hidden_dim, sigma, actor_lr, critic_lr, tau, gamma, device, env, 
                      batch_size=batch_size, buffer_size=buffer_size, sigma_min=sigma_min, 
                      decay_rate=decay_rate, policy_noise=policy_noise,
                      noise_clip=noise_clip, policy_freq=policy_freq)
    
    print("开始训练TD3算法...")
    print(f"设备: {device}")
    print(f"参数设置: M={M}, N={N}, K={K}, 所有灯都激活")
    print(f"训练回合数: {episodes}, 每回合步数: {max_steps}")
    print(f"批处理大小: {batch_size}, 缓冲区大小: {buffer_size}")
    print(f"学习率衰减: 初始Actor学习率={actor_lr}, Critic学习率={critic_lr}, 衰减率={lr_decay}, 频率={lr_decay_freq}回合")

    # 初始化状态回滚管理器（如果启用）
    rollback_manager = None
    enable_rollback = False
    if enable_rollback:
        rollback_manager = StateRollback(
            agent,
            checkpoint_dir = os.path.join(save_dir, 'checkpoints'),
            checkpoint_freq = checkpoint_freq,
            collapse_threshold = collapse_threshold,
            window_size = window_size,
            threshold_ratio = threshold_ratio
        )
        # 保存初始检查点
        rollback_manager.save_checkpoint(0, 0)
        print(f"状态回滚已启用: 检查点频率={checkpoint_freq}, 崩溃阈值={collapse_threshold}, 窗口大小={window_size}")

    # 保存训练配置参数
    save_config(args, save_dir)

    # 开始训练循环

    start_time = time.time()

    for i in range(episodes):
        if (i + 1) % 1 == 0:
            env.threshold = env.threshold - 0.8
            if env.threshold <= 0:
                env.threshold =0
        # 环境初始化
        state, _ = env.reset()
        episode_reward = 0
        episode_start_time = time.time()

        for step in range(max_steps):
            # 智能体选择动作
            action = agent.take_action(state, explore=True)
            
            # 环境执行动作
            reward, next_state, done = env.step(action)
            
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

        # 在每个回合结束后衰减噪声
        agent.sigma = agent.decay_noise()

        # 计算平均奖励和损失
        avg_score[i] = np.mean(ep_reward_list[-100:]) if len(ep_reward_list) > 0 else 0

        if len(ep_loss_list) > 0:
            avg_loss[i] = np.mean(ep_loss_list[-100:])
        else:
            avg_loss[i] = 0

        reward_diff = avg_score[i] - avg_score[i - 1]
        print('reward_diff', reward_diff)
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

        # 在每个回合结束后衰减噪声
        agent.decay_noise()
    
        # 学习率衰减（每lr_decay_freq个回合）
        if (i+1) % lr_decay_freq == 0:
            agent.decay_learning_rate(lr_decay, min_actor_lr, min_critic_lr)
            current_actor_lr, current_critic_lr = agent.get_learning_rates()
            print(f'学习率衰减: Actor学习率={current_actor_lr:.6f}, Critic学习率={current_critic_lr:.6f}\n')
        
        #print(f'回合 {i+1}, 损失 {avg_loss[i]:.4f}, 平均奖励 {avg_score[i]:.4f}, 本回合奖励 {episode_reward:.4f}')

        # 计算回合时间
        episode_time = time.time() - episode_start_time
        print(f'回合 {i + 1}/{episodes}, 耗时: {episode_time:.2f}秒, 损失: {avg_loss[i]:.4f}, '
                  f'平均奖励: {avg_score[i]:.4f}, 本回合奖励: {episode_reward:.4f}, '
                  f'噪声: {agent.sigma:.4f}\n')

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


        # 每50个回合保存一次训练曲线
        if (i+1) % 100 == 0:
            # 保存并绘制训练曲线
            plot_file = os.path.join(save_dir, f'training_curve_td3_ep{i+1}.png')
            plot_training_curve(range(i+1), avg_score[:i+1], plot_file)


            # 保存中间模型
            model_path = os.path.join(save_dir, f'td3_model_ep{i + 1}.pth')
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic1_state_dict': agent.critic1.state_dict(),
                'critic2_state_dict': agent.critic2.state_dict()
            }, model_path)
            print(f"TD3中间模型已保存到: {model_path}")
    # 训练结束后计算总时间
    total_time = time.time() - start_time
    print(f"训练完成! 总耗时: {total_time / 60:.2f}分钟")

    # 训练结束后绘制最终训练曲线
    final_plot_file = os.path.join(save_dir, '最终奖励_td3.png')
    plot_training_curve(range(episodes), avg_score, final_plot_file)
    
    # 保存 avg_score 和 avg_loss 数据
    np.save(os.path.join(save_dir, '平均奖励td3.npy'), avg_score)
    np.save(os.path.join(save_dir, '平均损失td3.npy'), avg_loss)

    # 保存训练好的模型
    model_path = os.path.join(save_dir, 'td3_model_final.pth')
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic1_state_dict': agent.critic1.state_dict(),
        'critic2_state_dict': agent.critic2.state_dict()
    }, model_path)
    print(f"TD3最终模型已保存到: {model_path}")

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

    plt.title('Training Performance Curve (TD3)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"训练曲线图已保存到: {filename}")


if __name__ == '__main__':
    # 打印Python和PyTorch版本信息
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")

    # 检查GPU状态
    print_gpu_memory_status()

    main()