from pathlib import Path
import random
import math
import time


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_distance_from(self, target):
        return  math.sqrt((self.x - target.x)**2 + (self.y - target.y)**2)
    
    @staticmethod
    def get_distance(city_1, city_2):
        return  math.sqrt((city_1.x - city_2.x)**2 + (city_1.y - city_2.y)**2)
    
    def __repr__(self):
        return f'{self.x}, {self.y}'


class Chromosome:
    def __init__(self, size, fill=False):
        self.size = size
        self.genes = []
        self.fitness = None
        self.selection_probability = None
        if fill:
            self.fill_chromosome()

    def fill_chromosome(self):
        self.genes = list(range(1, self.size))
        random.shuffle(self.genes)
        self.genes.insert(0, 0)


def get_cities(path):
    cities = []
    lines = path.read_text().splitlines()
    for line in lines:
        _, x, y = line.split()
        x, y = int(x), int(y)
        cities.append(City(x=x, y=y))
    return cities


MAP = get_cities(Path('data/TSP1002.txt'))


def get_path_length(path:Chromosome):
    length = 0
    for i, current_dest in enumerate(path.genes):
        next_dest = None
        if i < len(path.genes) - 1:
            next_dest = path.genes[i + 1]
        else:
            next_dest = path.genes[0]

        length += City.get_distance(MAP[current_dest], MAP[next_dest])

    return length


def get_chromosome_selection_probability(rank, population_size, s=2):
    return ((2 - s) / population_size) + ((2 * rank * (s - 1)) / (population_size * (population_size - 1)))


def mutate(path: Chromosome):
    mutated = Chromosome(path.size)
    mutated.genes = [0 for _ in range(mutated.size)]
    
    i, j = 0, 0
    while i == j:
        i = random.randint(0, path.size-1)
        j = random.randint(0, path.size-1)

    if i < j:
        mutated.genes[:i] = path.genes[:i]
        mutated.genes[i:j] = list(reversed(path.genes[i:j]))
        mutated.genes[j:] = path.genes[j:]
    else:
        mutated.genes[:j] = path.genes[:j]
        mutated.genes[j:i] = list(reversed(path.genes[j:i]))
        mutated.genes[i:] = path.genes[i:]

    first_city_index = mutated.genes.index(0)
    if first_city_index != 0:
        mutated.genes[first_city_index] = mutated.genes[0]
        mutated.genes[0] = 0

    mutated.fitness = get_path_length(mutated)
    return mutated


def de_mutation_for_tsp(A: Chromosome, B: Chromosome, C: Chromosome, F=0.5):
    size = A.size

    def get_edges(tour):
        return {(tour[i], tour[(i+1) % size]) for i in range(size)}

    edges_B = get_edges(B.genes)
    edges_C = get_edges(C.genes)

    edge_diff = list(edges_B - edges_C)

    if len(edge_diff) == 0:
        return mutate(A)

    num_to_use = max(1, min(len(edge_diff), int(F * len(edge_diff))))
    chosen_edges = random.sample(edge_diff, num_to_use)

    new_tour = A.genes[:]
    for (src, dst) in chosen_edges:
        try:
            i = new_tour.index(src)
            j = new_tour.index(dst)
            if (j != (i + 1) % size):
                new_tour.remove(dst)
                insert_pos = (new_tour.index(src) + 1) % len(new_tour)
                new_tour.insert(insert_pos, dst)
        except ValueError:
            continue

    mutated = Chromosome(size)
    mutated.genes = new_tour
    mutated.fitness = get_path_length(mutated)
    return mutated


def crossover(first_parent: Chromosome, second_parent: Chromosome):
    size = first_parent.size
    cut1 = random.randint(0, size - 2)
    cut2 = random.randint(cut1 + 1, size - 1)

    def pmx(p1, p2):
        child = [-1] * size

        child[cut1:cut2+1] = p2[cut1:cut2+1]

        mapping = {p2[i]: p1[i] for i in range(cut1, cut2+1)}

        for i in range(size):
            if i >= cut1 and i <= cut2:
                continue
            gene = p1[i]
            while gene in child[cut1:cut2+1]:
                gene = mapping[gene]
            child[i] = gene

        return child

    first_child = Chromosome(size)
    second_child = Chromosome(size)

    first_child.genes = pmx(first_parent.genes, second_parent.genes)
    second_child.genes = pmx(second_parent.genes, first_parent.genes)

    first_child.fitness = get_path_length(first_child)
    second_child.fitness = get_path_length(second_child)

    first_city_index = first_child.genes.index(0)
    if first_city_index != 0:
        first_child.genes[first_city_index] = first_child.genes[0]
        first_child.genes[0] = 0
    
    first_city_index = second_child.genes.index(0)
    if first_city_index != 0:
        second_child.genes[first_city_index] = second_child.genes[0]
        second_child.genes[0] = 0

    return first_child, second_child


n_paths = 10
n_matings = 20
n_iterations = 1000

start_time = time.time()

paths = [Chromosome(len(MAP), fill=True) for _ in range(n_paths)]
mating_pool = []

for path in paths:
    path.fitness = get_path_length(path)

for it in range(n_iterations):
    # Rank individuals based on their fitness and calculate cumulative probabilities
    ranked_paths = sorted(paths, key=lambda c: c.fitness)
    cumulative_probabilities = []
    cumulative_sum = 0
    for i, chromosome in enumerate(ranked_paths):
        prob = get_chromosome_selection_probability(i, n_paths, s=1.5)
        chromosome.selection_probability = prob
        cumulative_sum += prob
        cumulative_probabilities.append(cumulative_sum)

    # Select parents for mating pool using Roulette Wheel
    mating_pool.clear()
    for _ in range(n_matings):
        r = random.uniform(0, 1)
        for i, cp in enumerate(cumulative_probabilities):
            if cp >= r:
                mating_pool.append(ranked_paths[i])
                break
    
    # Apply cross-over over parents pairs
    pair_indexes = list(range(len(mating_pool)))
    random.shuffle(pair_indexes)
    i = 2
    while i <= len(pair_indexes):
        cross_over_indexes = pair_indexes[i-2:i]
        offsprings = crossover(mating_pool[cross_over_indexes[0]], mating_pool[cross_over_indexes[1]])
        mating_pool.append(offsprings[0])
        mating_pool.append(offsprings[1])
        i += 2

    # Apply mutations over individual parents
    for i in range(len(mating_pool)):
        A = mating_pool[i]
        B, C = random.sample([p for j, p in enumerate(mating_pool) if j != i], 2)
        mutated = de_mutation_for_tsp(A, B, C, F=0.5)
        if mutated.fitness < A.fitness:
            mating_pool[i] = mutated


    paths_mating_pool = (paths + mating_pool)
    paths_mating_pool = sorted(paths_mating_pool, key=lambda c: c.fitness)
    paths = paths_mating_pool[:n_paths]

    print(f'Iteration {it+1}')
    print(f'Best path length: {paths[0].fitness:.2f}\n')

print(f"Total execution time: {time.time() - start_time:.2f} s")