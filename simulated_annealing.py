import random
import numpy as np

from prediction import calculate_similarity


def simulated_annealing(history_numbers, max_iterations=1000, initial_temp=100, cooling_rate=0.95):
    def calculate_fitness(individual):
        similarities = [calculate_similarity(individual, history) for history in history_numbers]
        return 1 / (1 + max(similarities))  # 相似度越低，适应度越高

    def get_neighbor(individual):
        neighbor = individual.copy()
        if random.random() < 0.5:
            index = random.randint(0, 5)
            neighbor[index] = random.randint(1, 40)
        else:
            if random.random() < 0.5:
                neighbor[6] = random.randint(1, 40)
            else:
                neighbor[7] = random.randint(1, 10)
        return neighbor

    current_solution = create_individual()
    current_fitness = calculate_fitness(current_solution)
    best_solution = current_solution
    best_fitness = current_fitness
    temperature = initial_temp

    for iteration in range(max_iterations):
        neighbor = get_neighbor(current_solution)
        neighbor_fitness = calculate_fitness(neighbor)

        if neighbor_fitness > current_fitness:
            current_solution = neighbor
            current_fitness = neighbor_fitness
            if neighbor_fitness > best_fitness:
                best_solution = neighbor
                best_fitness = neighbor_fitness
        else:
            acceptance_probability = np.exp((neighbor_fitness - current_fitness) / temperature)
            if acceptance_probability > random.random():
                current_solution = neighbor
                current_fitness = neighbor_fitness

        temperature *= cooling_rate

    return best_solution


# 使用模拟退火算法优化预测结果
optimized_prediction = simulated_annealing(history_numbers)

# 打印优化后的预测结果
predicted_main_numbers = sorted(optimized_prediction[:6])  # 对主号码进行排序
predicted_bonus_number = optimized_prediction[6]
predicted_powerball_number = optimized_prediction[7]
print("Optimized Prediction using Simulated Annealing:")
print("Optimized Main Numbers:", predicted_main_numbers)
print("Optimized Bonus Number:", predicted_bonus_number)
print("Optimized Powerball Number:", predicted_powerball_number)
