import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from Validate import validate_lottery_numbers
from dibhttp import get_lotto_from_db

# 数据转换函数
def convert_data(datas):
    X_train, y_train, history_numbers = [], [], []
    for parts in datas:
        draw_id = parts[0]
        main_numbers = list(map(int, parts[2:8]))
        bonus_number = int(parts[8])
        powerball_number = int(parts[9])
        features = main_numbers
        label = main_numbers + [bonus_number, powerball_number]
        X_train.append(features)
        y_train.append(label)
        history_numbers.append(features + [bonus_number, powerball_number])
    return np.array(X_train), np.array(y_train), history_numbers

# 计算两个号码组合的相似度
def calculate_similarity(pred, history):
    common_elements = len(set(pred) & set(history))
    return common_elements / len(pred)

# 检查并修正号码范围
def check_and_fix_range(numbers, min_val, max_val):
    return [max(min(int(round(num)), max_val), min_val) for num in numbers]

# 贝叶斯更新函数
def bayesian_update(prior, data, n, index):
    likelihood = np.ones(n)
    for draw in data:
        likelihood[draw[index] - 1] += 1
    posterior = prior * likelihood
    posterior /= np.sum(posterior)
    return posterior

# 遗传算法部分
def create_individual():
    main_numbers = random.sample(range(1, 41), 6)
    bonus_number = random.randint(1, 40)
    powerball_number = random.randint(1, 10)
    return main_numbers + [bonus_number, powerball_number]

def calculate_fitness(individual, history_numbers):
    similarities = [calculate_similarity(individual, history) for history in history_numbers]
    return 1 / (1 + max(similarities))

def mutate(individual):
    if random.random() < 0.5:
        index = random.randint(0, 5)
        individual[index] = random.randint(1, 40)
    else:
        if random.random() < 0.5:
            individual[6] = random.randint(1, 40)
        else:
            individual[7] = random.randint(1, 10)
    return individual

def crossover(parent1, parent2):
    crossover_point = random.randint(1, 6)
    child1 = parent1[:crossover_point] + parent2[crossover_point:7] + parent1[7:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:7] + parent2[7:]
    return [child1, child2]

def genetic_algorithm(history_numbers, population_size=100, generations=100):
    population = [create_individual() for _ in range(population_size)]
    for _ in range(generations):
        population.sort(key=lambda ind: calculate_fitness(ind, history_numbers), reverse=True)
        next_generation = population[:population_size // 2]
        for i in range(population_size // 2):
            parents = random.sample(next_generation, 2)
            next_generation.extend(crossover(parents[0], parents[1]))
        population = [mutate(ind) for ind in next_generation]
    return sorted(population, key=lambda ind: calculate_fitness(ind, history_numbers), reverse=True)[:10]

# 检查并去重并确保范围合法
def remove_duplicates(prediction):
    main_numbers = prediction[:6]
    bonus_number = prediction[6]
    powerball_number = prediction[7]

    # 确保主号码不重复且在1到40范围内
    main_numbers = list(set(main_numbers))
    while len(main_numbers) < 6:
        new_number = random.randint(1, 40)
        if new_number not in main_numbers:
            main_numbers.append(new_number)
    main_numbers.sort()

    # 确保bonus_number在1到40范围内
    if bonus_number < 1 or bonus_number > 40:
        bonus_number = random.randint(1, 40)

    # 确保powerball_number在1到10范围内
    if powerball_number < 1 or powerball_number > 10:
        powerball_number = random.randint(1, 10)

    return main_numbers + [bonus_number, powerball_number]

# 模拟退火算法优化预测结果
def simulated_annealing(history_numbers, initial_state, temp, alpha, stopping_temp, max_iter):
    current_state = initial_state
    current_energy = calculate_fitness(current_state, history_numbers)
    best_state, best_energy = list(current_state), current_energy

    while temp > stopping_temp and max_iter > 0:
        max_iter -= 1
        candidate_state = list(current_state)
        index = random.randint(0, 7)
        if index < 6:
            candidate_state[index] = random.randint(1, 40)
        elif index == 6:
            candidate_state[index] = random.randint(1, 40)
        else:
            candidate_state[index] = random.randint(1, 10)
        candidate_state = remove_duplicates(candidate_state)
        is_valid, message = validate_lottery_numbers(candidate_state)
        if not is_valid:
            continue
        candidate_energy = calculate_fitness(candidate_state, history_numbers)
        energy_diff = candidate_energy - current_energy
        if energy_diff > 0 or random.uniform(0, 1) < np.exp(energy_diff / temp):
            current_state, current_energy = candidate_state, candidate_energy
        if current_energy > best_energy:
            best_state, best_energy = list(current_state), current_energy
        temp *= alpha

    return best_state

# 蚁群算法部分
class Ant:
    def __init__(self, num_main_numbers=6, num_bonus_numbers=1, num_powerball_numbers=1):
        self.num_main_numbers = num_main_numbers
        self.num_bonus_numbers = num_bonus_numbers
        self.num_powerball_numbers = num_powerball_numbers
        self.route = []

    def choose_number(self, pheromone, heuristic, alpha, beta):
        num_choices = len(pheromone)
        probabilities = [(pheromone[i] ** alpha) * (heuristic[i] ** beta) for i in range(num_choices)]
        probabilities /= np.sum(probabilities)
        chosen_number = np.random.choice(range(num_choices), p=probabilities)
        return chosen_number

    def build_route(self, pheromone_main, pheromone_bonus, pheromone_powerball, heuristic_main, heuristic_bonus, heuristic_powerball, alpha, beta):
        self.route = []

        for _ in range(self.num_main_numbers):
            chosen_number = self.choose_number(pheromone_main, heuristic_main, alpha, beta)
            self.route.append(chosen_number + 1)  # 假设号码从1开始

        bonus_number = self.choose_number(pheromone_bonus, heuristic_bonus, alpha, beta)
        self.route.append(bonus_number + 1)

        powerball_number = self.choose_number(pheromone_powerball, heuristic_powerball, alpha, beta)
        self.route.append(powerball_number + 1)

def update_pheromone(pheromone, ants, evaporation_rate, Q):
    pheromone *= (1 - evaporation_rate)
    for ant in ants:
        for number in ant.route:
            if number - 1 < len(pheromone):  # 确保索引合法
                pheromone[number - 1] += Q / len(ant.route)  # 假设号码从1开始

def ant_colony_optimization(history_numbers, num_ants=50, num_iterations=100, alpha=1.0, beta=1.0, evaporation_rate=0.5, Q=100, num_predictions=5):
    num_main_numbers = 40
    num_bonus_numbers = 40
    num_powerball_numbers = 10

    pheromone_main = np.ones(num_main_numbers)
    pheromone_bonus = np.ones(num_bonus_numbers)
    pheromone_powerball = np.ones(num_powerball_numbers)

    heuristic_main = np.ones(num_main_numbers)
    heuristic_bonus = np.ones(num_bonus_numbers)
    heuristic_powerball = np.ones(num_powerball_numbers)

    best_routes = []

    for _ in range(num_predictions):  # 进行num_predictions次运行以生成指定组数的结果
        best_route = None
        best_fitness = 0

        for _ in range(num_iterations):
            ants = [Ant() for _ in range(num_ants)]
            for ant in ants:
                ant.build_route(pheromone_main, pheromone_bonus, pheromone_powerball, heuristic_main, heuristic_bonus, heuristic_powerball, alpha, beta)
                ant.route = remove_duplicates(ant.route)
                valid, message = validate_lottery_numbers(ant.route)
                if not valid:
                    continue

                fitness = calculate_fitness(ant.route, history_numbers)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_route = list(ant.route)

            update_pheromone(pheromone_main, ants, evaporation_rate, Q)
            update_pheromone(pheromone_bonus, ants, evaporation_rate, Q)
            update_pheromone(pheromone_powerball, ants, evaporation_rate, Q)

        best_routes.append(best_route)

    return best_routes

# Example usage
data = get_lotto_from_db("select * from lotto order by id")
last_data = data[-1]
lastResult=last_data[2:10]
#data=data[0:len(data) - 1]

X_train, y_train, history_numbers = convert_data(data)
X_train = np.expand_dims(X_train, axis=1)

# 定义和训练LSTM模型
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=8))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# 使用模型进行预测
# 预测彩票号码
predictions = []
max_similarities = []
num_predictions = 10
while len(predictions) < num_predictions:
    # 增加随机性
    random_index = random.randint(0, len(X_train) - 1)
    predicted_numbers = model.predict(X_train[random_index].reshape(1, 1, X_train.shape[2]))[0]
    predicted_numbers = np.round(predicted_numbers).astype(int)

    predicted_numbers[:6] = check_and_fix_range(predicted_numbers[:6], 1, 40)
    predicted_numbers[6:7] = check_and_fix_range(predicted_numbers[6:7], 1, 40)
    predicted_numbers[7:] = check_and_fix_range(predicted_numbers[7:], 1, 10)

    predicted_numbers = remove_duplicates(predicted_numbers)

    valid, message = validate_lottery_numbers(predicted_numbers)
    max_similarity = max(calculate_similarity(predicted_numbers, history) for history in history_numbers)
    if valid and max_similarity < 0.6:
        # 计算预测值与历史数据的相似度最大值
        max_similarities.append(max_similarity)
        # 检查相似度，如果低于设定的阈值则添加到预测结果中
        predictions.append(predicted_numbers)

# 打印预测结果和相似度最大值
for i, prediction in enumerate(predictions):
    predicted_main_numbers = sorted(prediction[:6])
    predicted_bonus_number = prediction[6]
    predicted_powerball_number = prediction[7]
    print(f"Prediction {i + 1}:")
    print("Predicted Main Numbers:", predicted_main_numbers)
    print("Predicted Bonus Number:", predicted_bonus_number)
    print("Predicted Powerball Number:", predicted_powerball_number)
    valid, message = validate_lottery_numbers(prediction)
    print(valid, message)
    print("Max Similarity:", max_similarities[i])
    print()

# 初始化贝叶斯推理的先验概率
n_bonus, n_powerball = 40, 10
prior_bonus = np.ones(n_bonus) / n_bonus
prior_powerball = np.ones(n_powerball) / n_powerball

# 使用贝叶斯推理更新概率分布
posterior_bonus = bayesian_update(prior_bonus, history_numbers, n_bonus, 6)
posterior_powerball = bayesian_update(prior_powerball, history_numbers, n_powerball, 7)

# 打印更新后的概率分布
print("贝叶斯推理Bonus预测:")
for i in range(n_bonus):
    print(f"Number {i+1}: {posterior_bonus[i]:.4f}")
print("\n贝叶斯推理Powerball预测:")
for i in range(n_powerball):
    print(f"Number {i+1}: {posterior_powerball[i]: .4f}")

# 推荐最小概率的2个bonus和powerball号码
recommended_bonus = np.argsort(posterior_bonus)[:2] + 1
recommended_powerball = np.argsort(posterior_powerball)[:2] + 1
print("\n贝叶斯推理Bonus预测 最小可能概率:", recommended_bonus)
print("\n贝叶斯推理Powerball预测 最小可能概率:", recommended_powerball)

# 使用遗传算法优化预测结果
optimized_predictions = genetic_algorithm(history_numbers)
print("\n使用遗传算法优化预测结果:")
for i, prediction in enumerate(optimized_predictions):
    predicted_main_numbers = sorted(prediction[:6])
    predicted_bonus_number = prediction[6]
    predicted_powerball_number = prediction[7]
    print(f"Optimized Prediction {i + 1}:")
    print("Optimized Main Numbers:", predicted_main_numbers)
    print("Optimized Bonus Number:", predicted_bonus_number)
    print("Optimized Powerball Number:", predicted_powerball_number)

# 使用模拟退火算法进一步优化
initial_state = optimized_predictions[0]
temp, alpha, stopping_temp, max_iter = 1.0, 0.95, 0.001, 1000
final_state = simulated_annealing(history_numbers, initial_state, temp, alpha, stopping_temp, max_iter)
print("\n使用模拟退火算法优化结果:")
predicted_main_numbers = sorted(final_state[:6])
predicted_bonus_number = final_state[6]
predicted_powerball_number = final_state[7]
print("Simulated Annealing Main Numbers:", predicted_main_numbers)
print("Simulated Annealing Bonus Number:", predicted_bonus_number)
print("Simulated Annealing Powerball Number:", predicted_powerball_number)

# 使用蚁群算法进行彩票预测
aco_predictions = ant_colony_optimization(history_numbers, num_predictions=5)
print("\n使用蚁群算法预测结果:")
for i, prediction in enumerate(aco_predictions):
    predicted_main_numbers = sorted(prediction[:6])
    predicted_bonus_number = prediction[6]
    predicted_powerball_number = prediction[7]
    print(f"ACO Prediction {i + 1}:")
    print("ACO Main Numbers:", predicted_main_numbers)
    print("ACO Bonus Number:", predicted_bonus_number)
    print("ACO Powerball Number:", predicted_powerball_number)