import random
import csv
import time
from pathlib import Path

from utils.neural_network_loss import total_loss

def load_dataset(inputs_path='data/thyroidInputs.csv', targets_path='data/thyroidTargets.csv'):
    inputs = []
    targets = []

    path_inputs = Path(inputs_path)
    path_targets = Path(targets_path)

    if not path_inputs.exists() or not path_targets.exists():
        print("Warning: Dataset files not found. Creating dummy data.")
        Path('data').mkdir(exist_ok=True)
        with open(inputs_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for _ in range(100):
                writer.writerow([random.uniform(0, 1) for _ in range(21)])
        with open(targets_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for _ in range(100):
                row = ['0', '0', '0']
                row[random.randint(0, 2)] = '1'
                writer.writerow(row)


    lines_inputs = path_inputs.read_text(encoding='utf-8').splitlines()
    reader_inputs = csv.reader(lines_inputs)
    for row in reader_inputs:
        inputs.append([float(value) for value in row])

    lines_targets = path_targets.read_text(encoding='utf-8').splitlines()
    reader_targets = csv.reader(lines_targets)
    for row in reader_targets:
        targets.append(row.index('1'))

    return inputs, targets

class Chromosome:
    def __init__(self, size):
        self.size = size
        self.genes = []
        self.fitness = float('inf')

    def initialize_genes(self):
        self.genes = [random.uniform(-1, 1) for _ in range(self.size)]

def get_chromosome_fitness(chromosome, inputs, targets):
    return total_loss(chromosome.genes, inputs, targets)

def crossover(parent1, parent2, alpha=0.65):
    child1 = Chromosome(parent1.size)
    child2 = Chromosome(parent1.size)

    for i in range(parent1.size):
        gene1 = alpha * parent1.genes[i] + (1 - alpha) * parent2.genes[i]
        gene2 = alpha * parent2.genes[i] + (1 - alpha) * parent1.genes[i]
        child1.genes.append(min(max(gene1, -1), 1))
        child2.genes.append(min(max(gene2, -1), 1))

    return child1, child2

def mutate(chromosome, mutation_rate, mutation_strength):
    mutated_chromosome = Chromosome(chromosome.size)
    mutated_chromosome.genes = list(chromosome.genes)
    for i in range(chromosome.size):
        if random.random() < mutation_rate:
            mutation_value = random.uniform(-mutation_strength, mutation_strength)
            mutated_chromosome.genes[i] += mutation_value
            mutated_chromosome.genes[i] = max(-1, min(1, mutated_chromosome.genes[i]))
    return mutated_chromosome

ITERATIONS = 10
POPULATION_SIZE = 100
ELITISM_COUNT = 2
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.2
CROSSOVER_ALPHA = 0.65

inputs, targets = load_dataset()
GENE_COUNT = 158

start_time = time.time()

population = []
for _ in range(POPULATION_SIZE):
    chromosome = Chromosome(GENE_COUNT)
    chromosome.initialize_genes()
    chromosome.fitness = get_chromosome_fitness(chromosome, inputs, targets)
    population.append(chromosome)

best_solution_ever = None

for i in range(ITERATIONS):
    population.sort(key=lambda c: c.fitness)

    if best_solution_ever is None or population[0].fitness < best_solution_ever.fitness:
        best_solution_ever = population[0]

    print(f'Iteration {i+1:3} | Best Loss: {population[0].fitness:,.6f} | Best Ever: {best_solution_ever.fitness:,.6f}')

    next_generation = []
    next_generation.extend(population[:ELITISM_COUNT])

    while len(next_generation) < POPULATION_SIZE:
        tournament = random.sample(population, TOURNAMENT_SIZE)
        tournament.sort(key=lambda c: c.fitness)
        parent1 = tournament[0]
        
        tournament = random.sample(population, TOURNAMENT_SIZE)
        tournament.sort(key=lambda c: c.fitness)
        parent2 = tournament[0]

        child1, child2 = crossover(parent1, parent2, CROSSOVER_ALPHA)

        child1 = mutate(child1, MUTATION_RATE, MUTATION_STRENGTH)
        child2 = mutate(child2, MUTATION_RATE, MUTATION_STRENGTH)
        
        child1.fitness = get_chromosome_fitness(child1, inputs, targets)
        child2.fitness = get_chromosome_fitness(child2, inputs, targets)

        next_generation.append(child1)
        if len(next_generation) < POPULATION_SIZE:
            next_generation.append(child2)
    
    population = next_generation

execution_time = time.time() - start_time

print("\n--- Genetic Algorithm Finished ---")
print(f"Execution Time: {execution_time:.2f} seconds")
print(f"Best solution loss: {best_solution_ever.fitness:,.6f}")