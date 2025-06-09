import random
import time
import copy

class Chromosome:
    def __init__(self, size):
        self.size = size
        self.genes = []
        self.fitness = 0

    def initialize_genes(self):
        self.genes = list(range(1, self.size + 1))
        random.shuffle(self.genes)

def get_chromosome_fitness(chromosome):
    fitness = 0
    for i in range(chromosome.size):
        if chromosome.genes[i] == i + 1:
            fitness += 1
    return fitness

def crossover(parent1, parent2):
    size = parent1.size
    child1 = Chromosome(size)
    child2 = Chromosome(size)

    def pmx(p1_genes, p2_genes):
        child_genes = [-1] * size
        p1 = p1_genes[:]
        p2 = p2_genes[:]
        
        cut1, cut2 = sorted(random.sample(range(size), 2))

        child_genes[cut1:cut2+1] = p2[cut1:cut2+1]
        
        for i in range(cut1, cut2 + 1):
            if p1[i] not in child_genes:
                val_to_place = p1[i]
                p2_val_at_idx = p2[i]
                
                current_val = p2_val_at_idx
                while current_val in p1[cut1:cut2+1]:
                    idx = p1.index(current_val)
                    current_val = p2[idx]

                p1_idx = p1.index(p2_val_at_idx)
                
                j = p1_idx
                while child_genes[j] != -1:
                    j = p1.index(p2[j])
                    
                child_genes[j] = val_to_place

        for i in range(size):
            if child_genes[i] == -1:
                child_genes[i] = p1[i]
        return child_genes

    child1.genes = pmx(parent1.genes, parent2.genes)
    child2.genes = pmx(parent2.genes, parent1.genes)

    return child1, child2

def mutate(chromosome, mutation_rate):
    if random.random() > mutation_rate:
        return chromosome
    
    mutated_chromosome = Chromosome(chromosome.size)
    mutated_chromosome.genes = chromosome.genes[:]

    idx1, idx2 = sorted(random.sample(range(chromosome.size), 2))
    
    subsequence = mutated_chromosome.genes[idx1:idx2+1]
    subsequence.reverse()
    mutated_chromosome.genes[idx1:idx2+1] = subsequence
    
    return mutated_chromosome

ITERATIONS = 100
POPULATION_SIZE = 1200
GENE_COUNT = 100
ELITISM_COUNT = 2
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.1
CROSSOVER_CHANCE = 0.9

start_time = time.time()

population = []
for _ in range(POPULATION_SIZE):
    chromosome = Chromosome(GENE_COUNT)
    chromosome.initialize_genes()
    chromosome.fitness = get_chromosome_fitness(chromosome)
    population.append(chromosome)

best_solution_ever = None

for i in range(ITERATIONS):
    population.sort(key=lambda c: c.fitness, reverse=True)

    if best_solution_ever is None or population[0].fitness > best_solution_ever.fitness:
        best_solution_ever = copy.deepcopy(population[0])

    print(f'Iteration {i+1:3} | Best Fitness: {population[0].fitness:3} | Best Ever: {best_solution_ever.fitness:3}')

    if best_solution_ever.fitness == GENE_COUNT:
        print("\nOptimal solution found.")
        break

    next_generation = []
    next_generation.extend(population[:ELITISM_COUNT])

    while len(next_generation) < POPULATION_SIZE:
        
        tournament = random.sample(population, TOURNAMENT_SIZE)
        tournament.sort(key=lambda c: c.fitness, reverse=True)
        parent1 = tournament[0]

        if random.random() < CROSSOVER_CHANCE:
            tournament = random.sample(population, TOURNAMENT_SIZE)
            tournament.sort(key=lambda c: c.fitness, reverse=True)
            parent2 = tournament[0]
            child1, child2 = crossover(parent1, parent2)
        else:
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent1)

        child1 = mutate(child1, MUTATION_RATE)
        child2 = mutate(child2, MUTATION_RATE)
        
        child1.fitness = get_chromosome_fitness(child1)
        child2.fitness = get_chromosome_fitness(child2)

        next_generation.append(child1)
        if len(next_generation) < POPULATION_SIZE:
            next_generation.append(child2)
    
    population = next_generation

execution_time = time.time() - start_time

print("\n--- Genetic Algorithm Finished ---")
print(f"Execution Time: {execution_time:.2f} seconds")
print(f"Best solution fitness: {best_solution_ever.fitness} / {GENE_COUNT}")
print(f"Best solution genes: {best_solution_ever.genes}")