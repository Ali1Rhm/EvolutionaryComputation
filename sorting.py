import random
import time

class Chromosome:
    def __init__(self, size, fill=False):
        self.size = size
        self.genes = []
        self.fitness = None
        self.selection_probability = None
        if fill:
            self.fill_chromosome()

    def fill_chromosome(self):
        sorted_sequence = list(range(1, self.size+1))
        random.shuffle(sorted_sequence)
        self.genes = sorted_sequence

def get_chromosome_fitness(chromosome: Chromosome):
    fitness = 0
    for i, value in enumerate(chromosome.genes):
        if i+1 == value:
            fitness += 1
    return fitness

def get_chromosome_selection_probability(rank, population_size, s=2):
    return ((2 - s) / population_size) + ((2 * rank * (s - 1)) / (population_size * (population_size - 1)))

def cross_over(first_parent: Chromosome, second_parent: Chromosome):
    chromosome_size = first_parent.size

    first_child = Chromosome(chromosome_size)
    second_child = Chromosome(chromosome_size)
    first_cut_point = random.choice(range(0, chromosome_size-1))
    second_cut_point = first_cut_point + random.choice(range(1, chromosome_size-(first_cut_point)))

    first_child.genes = first_parent.genes[:]
    first_child.genes[first_cut_point:second_cut_point+1] = second_parent.genes[first_cut_point:second_cut_point+1]
    second_child.genes = second_parent.genes[:]
    second_child.genes[first_cut_point:second_cut_point+1] = first_parent.genes[first_cut_point:second_cut_point+1]

    for i in range(chromosome_size):
        if i >= first_cut_point and i <= second_cut_point:
            continue
        while first_child.genes.count(first_child.genes[i]) != 1:
            if i < first_cut_point:
                index = first_child.genes.index(first_child.genes[i], i+1)
                first_child.genes[i] = first_parent.genes[index]
            else:
                index = first_child.genes.index(first_child.genes[i])
                first_child.genes[i] = first_parent.genes[index]

        while second_child.genes.count(second_child.genes[i]) != 1:
            if i < first_cut_point:
                index = second_child.genes.index(second_child.genes[i], i+1)
                second_child.genes[i] = second_parent.genes[index]
            else:
                index = second_child.genes.index(second_child.genes[i])
                second_child.genes[i] = second_parent.genes[index]

    first_child.fitness = get_chromosome_fitness(first_child)
    second_child.fitness = get_chromosome_fitness(second_child)

    return first_child, second_child

def mutate(chromosome: Chromosome, mutation_rate = 0.8):
    mutated = Chromosome(chromosome.size)
    mutated.genes = chromosome.genes[:]
    mutated.fitness = chromosome.fitness
    p = random.uniform(0, 1)
    if p <= mutation_rate:
        return mutated
    
    first_cut_point = random.choice(range(0, chromosome.size-1))
    second_cut_point = first_cut_point + random.choice(range(1, chromosome.size-(first_cut_point)))

    slice = chromosome.genes[first_cut_point:second_cut_point+1]
    slice.reverse()

    mutated.genes[first_cut_point:second_cut_point+1] = slice
    mutated.fitness = get_chromosome_fitness(mutated)

    return mutated

iterations = 100
population_size = 400
mating_pool_size = 400
gene_count = 100
population = []
mating_pool = []

start_time = time.time()

current_iteration = 1
while current_iteration <= iterations:
    # Initialize population
    if current_iteration == 1:
        while len(population) < population_size:
            chromosome = Chromosome(size=gene_count, fill=True)
            chromosome.fitness = get_chromosome_fitness(chromosome)
            population.append(chromosome)
    
    # Rank individuals based on their fitness and calculate cumulative probabilities
    ranked_population = sorted(population, key=lambda c: c.fitness)
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
    pair_indexes = list(range(len(mating_pool)))
    random.shuffle(pair_indexes)
    i = 2
    while i <= len(pair_indexes):
        cross_over_indexes = pair_indexes[i-2:i]
        offsprings = cross_over(mating_pool[cross_over_indexes[0]], mating_pool[cross_over_indexes[1]])
        mating_pool.append(offsprings[0])
        mating_pool.append(offsprings[1])
        i += 2

    # Apply mutations over individual parents
    for chromosome in mating_pool:
        chromosome = mutate(chromosome)

    population_mating_pool = (population + mating_pool)
    population_mating_pool = sorted(population_mating_pool, key=lambda c: c.fitness)
    population = population_mating_pool[-population_size:]

    print(f'Iteration {current_iteration:}')
    print(f'Best chromosome fitness: {population[-1].fitness:_}\n')

    current_iteration += 1

print(f"\nExecution Time: {time.time() - start_time:.2f} seconds")