import random

import numpy as np

def bayesian_update(prior, data, n):
    likelihood = np.zeros(n)
    for draw in data:
        for number in draw:
            likelihood[number-1] += 1
    posterior = prior * likelihood
    posterior /= np.sum(posterior)
    return posterior

# 示例：使用贝叶斯推理更新概率分布
n = 49
prior = np.ones(n) / n
history = [[random.randint(1, 49) for _ in range(6)] for _ in range(100)]  # 假设100次历史开奖

posterior = bayesian_update(prior, history, n)

# 打印更新后的概率分布
for i in range(n):
    print(f"Number {i+1}: {posterior[i]:.4f}")
