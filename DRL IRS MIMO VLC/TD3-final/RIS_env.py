import random
import numpy as np

class Room_env:
    def __init__(self, M, N, K, mLED, nPD, kIRS):
        self.M = M  # 灯的个数
        self.N = N  # 用户数
        self.K = K  # ris个数
        self.mLED = mLED
        self.nPD = nPD
        self.kIRS = kIRS
        self.ris_loc = np.zeros((3, self.K))

    def init_ris_loc(self):

        # 大正方形的对角坐标
        min_point = np.array([0.5, 0.0, 0.5])
        max_point = np.array([2.5, 0.0, 2.5])

        # 计算大正方形的边长
        side_length = max_point[0] - min_point[0]  # 2.0

        # 每个小正方形的边长
        small_side = side_length / 2  # 1.0

        # 小正方形内点间距
        point_spacing = 0.2

        # 每个小正方形中点阵的行列数
        grid_size = self.kIRS  # 3x3网格

        # 计算点阵的总宽度 (3个点有2个间隔)
        grid_width = (grid_size - 1) * point_spacing  # 0.4

        # 计算边距（使点阵在小正方形中居中）
        margin = (small_side - grid_width) / 2  # (1.0 - 0.4)/2 = 0.3

        # 生成四个小正方形的左下角坐标
        small_squares = []
        for i in range(2):
            for j in range(2):
                x_start = min_point[0] + i * small_side
                z_start = min_point[2] + j * small_side
                small_squares.append((x_start, min_point[1], z_start))

        # 初始化三个数组分别存储x、y、z坐标
        x_coords = []
        y_coords = []
        z_coords = []

        # 生成所有点
        for square in small_squares:
            x_start, y, z_start = square
            for row in range(grid_size):
                for col in range(grid_size):
                    x = x_start + margin + col * point_spacing
                    z = z_start + margin + row * point_spacing

                    # 将坐标添加到各自的数组中
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(z)


        self.ris_loc[0, :] = np.array(x_coords)
        self.ris_loc[1, :] = np.array(y_coords)
        self.ris_loc[2, :] = np.array(z_coords)




        #x = np.linspace(1, 2.5, self.kIRS)  # 在 x 轴上 生成 ？ 个点
        #z = np.linspace(1, 2.5, self.kIRS)  # 在 z 轴上 生成 ？ 个点

        # 使用 meshgrid 生成网格
        #X, Z = np.meshgrid(x, z)
        #Y = np.zeros_like(X)  # y 坐标全为 0

        ## 将三维坐标组合在一起，并转置为 3×25 的矩阵
        #coordinates1 = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=0)
        #self.ris_loc = np.zeros((3, self.K))
        #self.ris_loc[0, :] = coordinates1[0, :]
        #self.ris_loc[1, :] = coordinates1[1, :]
        #self.ris_loc[2, :] = coordinates1[2, :]



        # for i in range(self.K):
        # self.ris_loc[:, i] = [3 + 0.25 * (i - int(i / 8) * 8), 0, int(i / 8) * 0.2 + 1.8]
        # x坐标

    def init_user_loc(self):
        self.user_loc = np.zeros((3, self.N))

        x = np.linspace(1, 2, self.nPD)
        y = np.linspace(1, 2, self.nPD)

        # 使用 meshgrid 生成网格
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)  # z 坐标全为 0

        # 将三维坐标组合在一起，并转置为 3×16 的矩阵
        coordinates2 = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=0)

        self.user_loc[0, :] = coordinates2[0, :]
        self.user_loc[1, :] = coordinates2[1, :]
        self.user_loc[2, :] = coordinates2[2, :]

    def init_LED_loc(self):
        # 创建 4×4 的网格点
        x = np.linspace(1, 2, self.mLED)
        y = np.linspace(1, 2, self.mLED)

        # 使用 meshgrid 生成网格
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, 3.5)

        # 将三维坐标组合在一起，并转置为 3×16 的矩阵
        coordinates3 = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=0)

        self.LED_loc = np.zeros((3, self.M))
        self.LED_loc[0, :] = coordinates3[0, :]
        self.LED_loc[1, :] = coordinates3[1, :]
        self.LED_loc[2, :] = coordinates3[2, :]

    def gen_Room(self):
        self.init_ris_loc()
        self.init_user_loc()
        self.init_LED_loc()


class RIS_env:
    def __init__(self, Room):
        self.Room = Room
        self.Room.gen_Room()

        # 就是self.action_dim最后加的K维度
        self.action_dim = self.Room.K+  self.Room.K * self.Room.N + self.Room.K * self.Room.M #+ self.Room.K
        self.zero_probability_dim = self.Room.K

        # 状态空间: G(K×M), F(K×N), A(M), r(K), H(N×M)
        self.state_dim = (self.Room.M + self.Room.N) * self.Room.K + self.Room.M + self.Room.K + self.Room.N * self.Room.M

        # 初始化信道增益矩阵
        self.h_direct = np.zeros((self.Room.N, self.Room.M))
        self.h_irs = np.zeros((self.Room.N, self.Room.M, self.Room.K))

        # 初始化系统参数
        self.f = np.zeros((self.Room.K, self.Room.N))
        self.g = np.zeros((self.Room.K, self.Room.M))
        self.zero_probability = np.zeros(self.zero_probability_dim)
        self.A = np.ones(self.Room.M) * 4.0
        self.r = np.zeros(self.Room.K)
        self.sigma_inv = np.zeros(self.Room.N)

        # 设置系统常数
        self.A_max = 4  # 最大光强
        self.alpha_ref = 0.9  # 最大反射系数
        self.noise_var = 1e-6
        self.irs_gain_factor = 1
        self.p = 0.5
        self.threshold = 0.8
        self.init_zero_probability = 0.4
        self.channel_scaling_factor = 1

    def get_action(self, action):
        """从动作值映射到具体操作并执行，直接确保满足约束"""
        for i in range(self.Room.K):
            if action[i] > self.r[i]:
                if action[i] < self.alpha_ref:
                    self.r[i] = action[i]
                else:
                    self.r[i] = self.alpha_ref
            else:
                self.r[i] = self.r[i]

        # 计算zero_probability，也就是F[k,n]G[k,m]为0的门限
        start_z = self.action_dim - self.zero_probability_dim
        end_z = self.action_dim
        action_store = action[start_z: end_z]
        self.zero_probability = 1 / (1 + np.exp(-action_store))

        # 将f从action中提取出来并使其满足行列分别相加总和各为1
        start_f = self.Room.K
        end_f = (self.Room.K + self.Room.K * self.Room.N)
        # 提取子数组
        sub_f = action[start_f: end_f]

        # 定义目标二维矩阵的行数和列数
        rows_f = self.Room.K
        cols_f = self.Room.M
        # 确保子数组的长度等于行列数的乘积
        if len(sub_f) != rows_f * cols_f:
            print("f的大小必须等于行列数的乘积")
            raise ValueError("子数组的长度必须等于行列数的乘积")
        # sub_f重组成二维矩阵
        two_f_matrix = sub_f.reshape(rows_f, cols_f)

        for a in range(self.Room.K):
            # 获取第i行的最大值索引
            max_idx_f = np.argmax(two_f_matrix[a, :])
            #if self.zero_probability[a] > self.threshold:
                # 将第i行的最大值位置设置为1
            self.f[a, :] = 0
            self.f[a, max_idx_f] = 1

        # 将g从action中提取出来并使其满足行列分别相加总和各为1
        start_g = (self.Room.K + self.Room.K * self.Room.N)  # 从索引  开始
        end_g = (self.Room.K + self.Room.K * self.Room.N + self.Room.K * self.Room.N)  # 到索引  结束
        # 提取子数组
        sub_g = action[start_g: end_g]
        # 定义目标二维矩阵的行数和列数
        rows_g = self.Room.K
        cols_g = self.Room.M
        # 确保子数组的长度等于行列数的乘积
        if len(sub_g) != rows_g * cols_g:
            print("g 的大小必须等于行列数的乘积")
            raise ValueError("子数组的长度必须等于行列数的乘积")
        # 重组成二维矩阵
        two_g_matrix = sub_g.reshape(rows_g, cols_g)

        for b in range(self.Room.K):
            # 获取第i行的最大值索引
            max_idx_g = np.argmax(two_g_matrix[b, :])

            self.g[b, :] = 0
            self.g[b, max_idx_g] = 1

    def calculate_channel_gains(self):
        """计算直射和IRS反射信道增益"""
        # 计算距离
        d_LED_to_ris = np.zeros((self.Room.K, self.Room.M))
        d_ris_to_user = np.zeros((self.Room.K, self.Room.N))
        d_LED_to_user = np.zeros((self.Room.M, self.Room.N))

        for m in range(self.Room.M):
            for n in range(self.Room.N):
                d_LED_to_user[m, n] = np.sqrt(np.sum((self.Room.LED_loc[:, m] - self.Room.user_loc[:, n]) ** 2))

        for k in range(self.Room.K):
            for m in range(self.Room.M):
                d_LED_to_ris[k, m] = np.sqrt(np.sum((self.Room.ris_loc[:, k] - self.Room.LED_loc[:, m]) ** 2))
            for n in range(self.Room.N):
                d_ris_to_user[k, n] = np.sqrt(np.sum((self.Room.ris_loc[:, k] - self.Room.user_loc[:, n]) ** 2))

        # 计算角度
        # los_theta：LED与PD之间的辐射角,也是入射角
        los_theta = np.zeros((self.Room.M, self.Room.N))
        for m in range(self.Room.M):
            for n in range(self.Room.N):
                los_theta[m, n] = np.arccos((self.Room.LED_loc[2, m] - self.Room.user_loc[2, n]) / d_LED_to_user[m, n])
        # print('los_theta\n',los_theta)

        # nlos_theta_S：LED到IRS的入射角
        nlos_theta_S = np.zeros((self.Room.K, self.Room.M))
        for k in range(self.Room.K):
            for m in range(self.Room.M):
                nlos_theta_S[k, m] = np.arccos((self.Room.LED_loc[2, m] - self.Room.ris_loc[2, k]) / d_LED_to_ris[k, m])

        # nlos_theta_D: IRS到PD的辐射角
        nlos_theta_D = np.zeros((self.Room.K, self.Room.N))
        for k in range(self.Room.K):
            for n in range(self.Room.N):
                nlos_theta_D[k, n] = np.arccos(
                    (self.Room.ris_loc[2, k] - self.Room.user_loc[2, n]) / d_ris_to_user[k, n])

        # 设置光通信参数
        A_p = 0.001  # 接收PD的探测面积
        Lam_index_k = 1  # 光源朗伯辐射阶数
        fai_c = np.radians(70)  # 半视场角
        g_fai = 1  # 光学滤波器增益
        n_r = 1.5  # 折射率

        # 计算直射信道增益 h_los
        for m in range(self.Room.M):
            for n in range(self.Room.N):
                if los_theta[m, n] <= fai_c and los_theta[m, n] >= 0:
                    T_fai = n_r ** 2 / (np.sin(fai_c) ** 2)
                else:
                    T_fai = 0  # 计算光学集中器增益

                self.h_direct[n, m] = ((A_p * (Lam_index_k + 1) / (2 * np.pi * (d_LED_to_user[m, n]) ** 2))
                                       * np.power(np.cos(los_theta[m, n]), Lam_index_k)
                                       * T_fai * g_fai * np.cos(los_theta[m, n]))  # np.power计算次方

        # 计算IRS反射信道增益 h_irs
        for n in range(self.Room.N):
            for m in range(self.Room.M):
                for k in range(self.Room.K):
                    # 计算角度相关项
                    cos_phi_S = np.cos(nlos_theta_S[k, m])
                    cos_phi_D = np.cos(nlos_theta_D[k, n])

                    # 根据视场角计算T函数
                    if nlos_theta_S[k, m] <= fai_c and nlos_theta_D[k, n] <= fai_c:
                        T_phi_S = (n_r ** 2) / (np.sin(fai_c) ** 2)

                    else:
                        T_phi_S = 0

                    # 计算IRS的单位法向量(N_k^IRS)
                    N_k_irs = np.array([0, 1, 0])  # IRS的单位法向量

                    # 计算实际的RIS到用户的方向向量
                    ris_to_user_vec = self.Room.user_loc[:, n] - self.Room.ris_loc[:, k]
                    E_kn_RD = ris_to_user_vec / np.linalg.norm(ris_to_user_vec)

                    # 计算内积(N_k^IRS)^T·E_k,n^RD
                    N_dot_E = np.dot(N_k_irs, E_kn_RD)

                    # 计算反射信道增益
                    d_total = d_LED_to_ris[k, m] + d_ris_to_user[k, n]
                    self.h_irs[n, m, k] = ((A_p * (Lam_index_k + 1) * cos_phi_S ** (Lam_index_k) * cos_phi_D) /
                                           (2 * np.pi * (d_total) ** 2)) * T_phi_S * g_fai * N_dot_E

    def compute_H_matrix(self):
        """计算信道矩阵H"""
        # 基础噪声水平
        sigma_base = self.noise_var
        # 使用统一的噪声水平
        sigma = np.full(self.Room.N, sigma_base)

        # 构建噪声协方差矩阵
        sigma_matrix = np.diag(sigma)
        # 构建信道矩阵
        H = np.zeros((self.Room.N, self.Room.M))

        # 计算缩放因子
        s_f = np.sum(self.h_direct ** 2)
        if s_f > 0:
            scaling_factor = 1.0 / np.sqrt(s_f)
            # 保存归一化因子
            self.channel_scaling_factor = scaling_factor
        else:
            print("缩放因子有误")
            raise ValueError("缩放因子有误")

        # 计算噪声归一化之后的噪声协方差矩阵
        self.sigma_inv = np.linalg.inv(sigma_matrix) * self.channel_scaling_factor

        # 计算H = h^(D) + (f ⊙ g)^T (r^T ⊙ h)
        for n in range(self.Room.N):
            for m in range(self.Room.M):
                # 直射部分
                H[n, m] = (self.h_direct[n, m]
                           * self.channel_scaling_factor)  # * self.irs_gain_factor

                # IRS反射部分
                for k in range(self.Room.K):
                    # 按照公式计算
                    H[n, m] += (self.f[k, n] * self.g[k, m] * self.r[k] * self.h_irs[n, m, k]
                                * self.channel_scaling_factor * self.irs_gain_factor)

        return H

    def compute_capacity(self, H):

        try:
            H_H = H.T
            HTH = H_H @ self.sigma_inv @ H

            eigenvalues = np.linalg.eigvalsh(HTH)
            if np.any(eigenvalues <= 0):
                print('zhizhihzihzihizzhi')
                return -10

            # 使用特征值乘积计算行列式
            det_term = np.prod(eigenvalues)

            # 计算log_det_term
            log_det_term = 0.5 * np.log10(det_term)

            # 计算∑log(A_m)
            log_A_sum = np.sum(np.log10(np.maximum(self.A, 1e-10)))

            # 总容量
            capacity = log_det_term + log_A_sum
            return capacity
        except:
            return -10

    def step(self, action):
        """执行动作并转移状态"""
        # 保存当前状态作为比较
        old_state = self.get_state()

        # 解析并执行动作
        self.get_action(action)

        # 计算信道增益
        self.calculate_channel_gains()

        # 计算信道矩阵H
        H = self.compute_H_matrix()

        # 计算系统容量
        capacity = self.compute_capacity(H)

        # 获取新状态
        new_state = self.get_state()

        # 计算奖励
        reward = capacity

        done = False

        return reward, new_state, done

    def get_state(self):
        """获取当前系统状态"""
        # 计算信道矩阵H
        H = self.compute_H_matrix()

        # 状态由vec(G), vec(F), vec(A), vec(r), vec(H)组成
        g_flat = self.g.flatten()
        f_flat = self.f.flatten()
        H_flat = H.flatten()

        # 组合成状态向量
        state = np.concatenate([g_flat, f_flat, self.A, self.r, H_flat])
        return state

    def reset(self):
        """重置环境状态"""
        # 初始化系统参数
        self.f = np.zeros((self.Room.K, self.Room.N))
        self.g = np.zeros((self.Room.K, self.Room.M))
        self.A = np.ones(self.Room.M) * self.A_max
        self.r = np.zeros(self.Room.K)
        self.zero_probability = np.ones(self.Room.K) * self.init_zero_probability

        # 随机初始化一些有效配置
        # 为每个IRS随机分配灯和用户
        for k in range(self.Room.K):
            # 每个IRS连接到一个灯
            suijishu = random.random()
            led_idx = np.random.choice(self.Room.M)
            self.g[k, led_idx] = 1
            if suijishu < self.p:
                self.g[k, :] = 0

            # 每个IRS连接到一个用户
            user_idx = np.random.choice(self.Room.N)
            self.f[k, user_idx] = 1
            if suijishu < self.p:
                self.f[k, :] = 0

            # 设置随机反射系数
            self.r[k] = np.random.uniform(0, 0.9)

        self.calculate_channel_gains()

        # 获取初始状态
        init_state = self.get_state()

        # 计算初始容量
        H = self.compute_H_matrix()
        init_capacity = self.compute_capacity(H)

        return init_state, init_capacity 