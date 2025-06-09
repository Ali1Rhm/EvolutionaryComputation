import random
import csv
import time
from pathlib import Path

def load_dataset(file_path='data/knapsak.csv'):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: The file '{file_path}' was not found.")
        exit()
    lines = path.read_text().splitlines()
    reader = csv.reader(lines)
    next(reader)

    weights = []
    values = []

    for row in reader:
        weights.append(int(row[0]))
        values.append(int(row[1]))
    
    return weights, values

class Chromosome:
    def __init__(self, size):
        self.size = size
        self.genes = []
        self.fitness = 0

    def initialize_genes(self):
        self.genes = [random.choice([0, 1]) for _ in range(self.size)]

def get_chromosome_fitness(chromosome: Chromosome, weights: list, values: list, max_weight: int) -> int:
    current_weight = 0
    current_value = 0
    for i in range(chromosome.size):
        if chromosome.genes[i] == 1:
            current_weight += weights[i]
            current_value += values[i]
    
    if current_weight > max_weight:
        return 0
    return current_value

def get_chromosome_weight(chromosome: Chromosome, weights: list) -> int:
    weight = 0
    for i in range(chromosome.size):
        weight += weights[i] * chromosome.genes[i]
    return weight

def crossover(parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
    child1 = Chromosome(parent1.size)
    child2 = Chromosome(parent1.size)
    for i in range(parent1.size):
        if random.random() < 0.5:
            child1.genes.append(parent1.genes[i])
            child2.genes.append(parent2.genes[i])
        else:
            child1.genes.append(parent2.genes[i])
            child2.genes.append(parent1.genes[i])
    return child1, child2

def mutate(chromosome: Chromosome, mutation_rate: float) -> Chromosome:
    mutated_chromosome = Chromosome(chromosome.size)
    mutated_chromosome.genes = list(chromosome.genes)
    for i in range(chromosome.size):
        if random.random() < mutation_rate:
            mutated_chromosome.genes[i] = 1 - mutated_chromosome.genes[i]
    return mutated_chromosome

MAX_WEIGHT = 6_404_180
ITERATIONS = 200
POPULATION_SIZE = 100
ELITISM_COUNT = 2
MUTATION_RATE = 0.02

weights, values = load_dataset()
GENE_COUNT = len(weights)

start_time = time.time()

population = []
while len(population) < POPULATION_SIZE:
    chromosome = Chromosome(size=GENE_COUNT)
    chromosome.initialize_genes()
    chromosome.fitness = get_chromosome_fitness(chromosome, weights, values, MAX_WEIGHT)
    if chromosome.fitness > 0:
        population.append(chromosome)

best_solution_ever = None

for i in range(ITERATIONS):
    population.sort(key=lambda c: c.fitness, reverse=True)

    if best_solution_ever is None or population[0].fitness > best_solution_ever.fitness:
        best_solution_ever = population[0]

    print(f'Iteration {i+1:3} | Best Fitness: {population[0].fitness:_} | Best Ever: {best_solution_ever.fitness:_}')

    next_generation = []

    next_generation.extend(population[:ELITISM_COUNT])

    while len(next_generation) < POPULATION_SIZE:
        parent1 = random.choice(population[:POPULATION_SIZE // 2])
        parent2 = random.choice(population[:POPULATION_SIZE // 2])

        child1, child2 = crossover(parent1, parent2)

        child1 = mutate(child1, MUTATION_RATE)
        child2 = mutate(child2, MUTATION_RATE)

        child1.fitness = get_chromosome_fitness(child1, weights, values, MAX_WEIGHT)
        child2.fitness = get_chromosome_fitness(child2, weights, values, MAX_WEIGHT)
        
        if child1.fitness > 0:
            next_generation.append(child1)
        if len(next_generation) < POPULATION_SIZE and child2.fitness > 0:
            next_generation.append(child2)
    
    population = next_generation

final_best_chromosome = best_solution_ever
final_weight = get_chromosome_weight(final_best_chromosome, weights)
execution_time = time.time() - start_time

print("\n--- Genetic Algorithm Finished ---")
print(f"Execution Time: {execution_time:.2f} seconds")
print(f"Best solution fitness: {final_best_chromosome.fitness:_}")
print(f"Best solution weight: {final_weight:_} / {MAX_WEIGHT:_}")
print(f"Best solution genes: {final_best_chromosome.genes}")