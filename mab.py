import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, epsilon, num_arms):
        self.epsilon = epsilon
        self.num_arms = num_arms
        self.arm_values = np.random.normal(0, 1, num_arms)  # 每个臂的真实奖励值
        self.est_values = np.zeros(num_arms)  # 每个臂的估计奖励值
        self.arm_counts = np.zeros(num_arms)  # 每个臂的选择次数

    def choose_arm(self):
        if np.random.rand() < self.epsilon:
            # 以概率 epsilon 进行探索，选择一个随机臂
            return np.random.choice(self.num_arms)
        else:
            # 以概率 1-epsilon 进行利用，选择当前估计奖励值最高的臂
            return np.argmax(self.est_values)

    def update_arm(self, chosen_arm, reward):
        # 更新选择的臂的估计奖励值和选择次数
        self.arm_counts[chosen_arm] += 1
        self.est_values[chosen_arm] += (reward - self.est_values[chosen_arm]) / self.arm_counts[chosen_arm]

class UCBBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.arm_values = np.random.normal(0, 1, num_arms)  # 每个臂的真实奖励值
        self.est_values = np.zeros(num_arms)  # 每个臂的估计奖励值
        self.arm_counts = np.zeros(num_arms)  # 每个臂的选择次数

    def choose_arm(self):
        # 使用 UCB 算法选择臂
        total_counts = np.sum(self.arm_counts)
        ucb_values = self.est_values + np.sqrt(2 * np.log(total_counts + 1) / (self.arm_counts + 1e-6))
        return np.argmax(ucb_values)

    def update_arm(self, chosen_arm, reward):
        # 更新选择的臂的估计奖励值和选择次数
        self.arm_counts[chosen_arm] += 1
        self.est_values[chosen_arm] += (reward - self.est_values[chosen_arm]) / self.arm_counts[chosen_arm]

class ThompsonSamplingBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.arm_values = np.random.normal(0, 1, num_arms)  # 每个臂的真实奖励值
        self.alpha = np.ones(num_arms)  # 每个臂的正态分布参数的先验 alpha
        self.beta = np.ones(num_arms)  # 每个臂的正态分布参数的先验 beta

    def choose_arm(self):
        # 使用 Thompson Sampling 算法选择臂
        sampled_values = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_values)

    def update_arm(self, chosen_arm, reward):
        # 更新选择的臂的正态分布参数的先验
        if reward == 1:
            self.alpha[chosen_arm] += 1
        else:
            self.beta[chosen_arm] += 1

if __name__ == "__main__":
    num_arms = 2
    ts_bandit = ThompsonSamplingBandit(num_arms=num_arms)

    ucb_bandit = UCBBandit(num_arms=num_arms)

    epsilon_value = 0.1
    bandit = EpsilonGreedyBandit(epsilon=epsilon_value, num_arms=num_arms)

    num_iterations = 1000
    total_reward = 0

    for _ in range(num_iterations):
        chosen_arm = bandit.choose_arm()
        
        # 模拟真实奖励，这里使用标准正态分布
        reward = np.random.normal(bandit.arm_values[chosen_arm], 1)
        
        total_reward += reward
        bandit.update_arm(chosen_arm, reward)

    average_reward = total_reward / num_iterations
    print(f"Average Reward: {average_reward}")
    print(f"Estimated Arm Values: {bandit.est_values}")