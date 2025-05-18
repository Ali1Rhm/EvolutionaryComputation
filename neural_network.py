from pathlib import Path
import csv
import time
import random

from utils.neural_network_loss import total_loss

def load_dataset():
    inputs = []
    targets = []

    path_inputs = Path('data/thyroidInputs.csv')
    path_targets = Path('data/thyroidTargets.csv')

    lines = path_inputs.read_text(encoding='utf-8').splitlines()
    reader = csv.reader(lines)
    for row in reader:
        input = [float(value) for value in row]
        inputs.append(input)

    lines = path_targets.read_text(encoding='utf-8').splitlines()
    reader = csv.reader(lines)
    for row in reader:
        target = row.index('1')
        targets.append(target)

    return inputs, targets

class Chromosome:
    def __init__(self, size, fill=False):
        self.size = size
        self.genes = []
        self.fitness = None
        self.selection_probability = None
        if fill:
            self.fill_chromosome()

    def fill_chromosome(self):
        self.genes = [random.uniform(-1, 1) for _ in range(self.size)]

def get_chromosome_fitness(chromosome: Chromosome):
    return total_loss(chromosome.genes, inputs, targets)

def get_chromosome_selection_probability(rank, population_size, s=2):
    return ((2 - s) / population_size) + ((2 * rank * (s - 1)) / (population_size * (population_size - 1)))

def crossover(first_parent: Chromosome, second_parent: Chromosome, a=0.5):
    size = first_parent.size
    first_child = Chromosome(size)
    second_child = Chromosome(size)

    for i in range(size):
        g1 = a * first_parent.genes[i] + (1 - a) * second_parent.genes[i]
        g2 = a * second_parent.genes[i] + (1 - a) * first_parent.genes[i]
        first_child.genes.append(min(max(g1, -1), 1))
        second_child.genes.append(min(max(g2, -1), 1))

    first_child.fitness = get_chromosome_fitness(first_child)
    second_child.fitness = get_chromosome_fitness(second_child)

    return first_child, second_child

def mutate(chromosome: Chromosome, mutation_rate=0.1, mutation_strength=0.1):
    for i in range(chromosome.size):
        if random.random() < mutation_rate:
            chromosome.genes[i] += random.uniform(-mutation_strength, mutation_strength)
            chromosome.genes[i] = max(-1, min(1, chromosome.genes[i]))
    chromosome.fitness = get_chromosome_fitness(chromosome)
    return chromosome

inputs, targets = load_dataset()
iterations = 20
population_size = 10
mating_pool_size = 10
gene_count = 158
population = []
mating_pool = []

start_time = time.time()

for current_iteration in range(1, iterations + 1):
    # Initialize population
    if current_iteration == 1:
        for _ in range(population_size):
            chromosome = Chromosome(gene_count, fill=True)
            chromosome.fitness = get_chromosome_fitness(chromosome)
            population.append(chromosome)

    # Rank individuals based on their fitness and calculate cumulative probabilities
    ranked_population = sorted(population, reverse=True, key=lambda c: c.fitness)
    cumulative_probabilities = []
    cumulative_sum = 0
    for i, chromosome in enumerate(ranked_population):
        prob = get_chromosome_selection_probability(i, population_size, s=1.5)
        chromosome.selection_probability = prob
        cumulative_sum += prob
        cumulative_probabilities.append(cumulative_sum)

    # Select parents for mating pool using Roulette Wheel
    mating_pool.clear()
    for _ in range(mating_pool_size):
        r = random.uniform(0, 1)
        for i, cp in enumerate(cumulative_probabilities):
            if cp >= r:
                mating_pool.append(ranked_population[i])
                break

    # Apply cross-over over parents pairs
    random.shuffle(mating_pool)
    offspring = []
    for i in range(0, len(mating_pool) - 1, 2):
        c1, c2 = crossover(mating_pool[i], mating_pool[i + 1], a=0.65)
        offspring.extend([c1, c2])

    # Apply mutations over individual parents
    for i in range(len(offspring)):
        offspring[i] = mutate(offspring[i], mutation_rate=1, mutation_strength=0.2)

    combined = population + offspring
    combined.sort(reverse=True, key=lambda c: c.fitness)
    population = combined[-population_size:]

    print(f"Iteration {current_iteration}")
    print(f"Best fitness: {population[0].fitness:.6f}\n")

print(f"\nExecution Time: {time.time() - start_time:.2f} seconds")