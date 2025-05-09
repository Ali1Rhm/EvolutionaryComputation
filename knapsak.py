import random
import csv
import time
from pathlib import Path

start_time = time.time()

MAX_WEIGHT = 6_404_180
WEIGHTS = []
VALUES = []

path = Path('knapsak.csv')
lines = path.read_text().splitlines()
reader = csv.reader(lines)
headers_row = next(reader)

for row in reader:
    WEIGHTS.append(int(row[0]))
    VALUES.append(int(row[1]))

class Chromosome:
    def __init__(self, size, fill=False):
        self.size = size
        self.genes = []
        self.fitness = None
        self.selection_probability = None
        if fill:
            self.fill_chromosome()

    def fill_chromosome(self):
        while(len(self.genes) < self.size):
            self.genes.append(random.choice([0, 1]))

def validate_chromosome(chromosome: Chromosome):    
    return get_chromosome_weight(chromosome) <= MAX_WEIGHT

def get_chromosome_weight(chromosome: Chromosome):
    weight = 0
    for i in range(chromosome.size):
        weight += WEIGHTS[i] * chromosome.genes[i]
    return weight

def get_chromosome_value(chromosome: Chromosome):
    value = 0
    for i in range(chromosome.size):
        value += VALUES[i] * chromosome.genes[i]
    return value

def get_chromosome_fitness(chromosome: Chromosome):
    fitness = 0

    if get_chromosome_weight(chromosome) > MAX_WEIGHT:
        return fitness
    
    for i in range(chromosome.size):
        fitness += VALUES[i] * WEIGHTS[i] * chromosome.genes[i]
    return fitness

def get_chromosome_selection_probability(rank, population_size, s=2):
    return ((2-s)/population_size) + ((2*rank*(s - 1)) / (population_size*(population_size-1)))

# Uniform
def cross_over(first_parent: Chromosome, second_parent: Chromosome, p=0.5):
    chromosome_size = first_parent.size
    first_child = Chromosome(chromosome_size)
    second_child = Chromosome(chromosome_size)
    for i in range(chromosome_size):
        l = random.uniform(0, 1)
        if l >= 0.5:
            first_child.genes.append(first_parent.genes[i])
            second_child.genes.append(second_parent.genes[i])
        else:
            first_child.genes.append(second_parent.genes[i])
            second_child.genes.append(first_parent.genes[i])
    first_child.fitness = get_chromosome_fitness(first_child)
    second_child.fitness = get_chromosome_fitness(second_child)
    return (first_child, second_child)

def mutate(chromosome: Chromosome, mutation_rate = 0.8):
    chromosome_size = chromosome.size
    mutated = Chromosome(chromosome.size)
    for i in range(chromosome_size):
        l = random.uniform(0, 1)
        if l > mutation_rate:
            mutated.genes.append(-1 * (chromosome.genes[i] - 1))
        else:
            mutated.genes.append(chromosome.genes[i])
    mutated.fitness = get_chromosome_fitness(mutated)
    return mutated

iterations = 100
population_size = 10
mating_pool_size = 20
population = []
mating_pool = []

current_iteration = 1
while current_iteration <= iterations:
    # Initialize population
    if current_iteration == 1:
        while(len(population) < population_size):
            chromosome = Chromosome(24, fill=True)
            if validate_chromosome(chromosome) != True:
                continue
            chromosome.fitness = get_chromosome_fitness(chromosome)
            population.append(chromosome)

    # Rank individuals based on their fitness and calculate cumulative probabilities
    ranked_population = sorted(population, key=lambda c: c.fitness)
    cumulative_probabilities = 0
    a = []
    for i, chromosome in enumerate(ranked_population):
        chromosome.selection_probability = get_chromosome_selection_probability(i, population_size)
        cumulative_probabilities += chromosome.selection_probability
        a.append(cumulative_probabilities)
    
    # Select parents for mating pool using Roulette Wheel
    current_member = 1
    mating_pool.clear()
    while current_member <= mating_pool_size:
        r = random.uniform(0, 1)
        i = 0
        while a[i] < r:
            i += 1
        mating_pool.append(ranked_population[i])
        current_member += 1

    # Apply cross-over over parents pairs
    pair_indexes = list(range(len(mating_pool)))
    random.shuffle(pair_indexes)
    i = 2
    while i <= len(pair_indexes):
        offsprings = cross_over(mating_pool[0], mating_pool[1])
        mating_pool.append(offsprings[0])
        mating_pool.append(offsprings[1])

        cross_over_indexes = pair_indexes[i-2:i]
        i += 2

    # Apply mutations over individual parents
    for chromosome in mating_pool:
        chromosome = mutate(chromosome)

    population_mating_pool = (population + mating_pool)
    population_mating_pool = sorted(population_mating_pool, key=lambda c: c.fitness)
    population = population_mating_pool[-population_size:]

    print(f'Iteration {current_iteration:}')
    print(f'Best chromosome: {population[-1].genes} | {population[-1].fitness} | {MAX_WEIGHT - get_chromosome_weight(population[-1])}\n')

    current_iteration += 1

print(f'\nExecution Time: {time.time() - start_time}')