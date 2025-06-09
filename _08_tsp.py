import random
import math
import time
from pathlib import Path

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

class Chromosome:
    def __init__(self, dim, genes=None):
        self.dim = dim
        if genes is None:
            self.genes = [random.random() for _ in range(dim)]
        else:
            self.genes = genes
        self.fitness = None

    def decode(self):
        return sorted(range(self.dim), key=lambda i: self.genes[i])

    def evaluate(self, cities):
        tour = self.decode()
        total = 0.0
        for i in range(self.dim):
            a = cities[tour[i]]
            b = cities[tour[(i + 1) % self.dim]]
            total += City.distance(a, b)
        self.fitness = total
        return total

def load_cities(path):
    cities = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            _, x, y = parts[:3]
            cities.append(City(float(x), float(y)))
    return cities

def mutate(pop, idx, F):
    idxs = list(range(len(pop)))
    idxs.remove(idx)
    a, b, c = random.sample(idxs, 3)
    A = pop[a].genes
    B = pop[b].genes
    C = pop[c].genes
    return [A[i] + F * (B[i] - C[i]) for i in range(len(A))]

def crossover(target, mutant, CR):
    dim = len(target)
    trial = [0.0] * dim
    jrand = random.randrange(dim)
    for j in range(dim):
        if random.random() < CR or j == jrand:
            trial[j] = mutant[j]
        else:
            trial[j] = target[j]
    return [min(max(v, 0.0), 1.0) for v in trial]

def differential_evolution_tsp(cities, pop_size=50, F=0.8, CR=0.9, generations=1000):
    dim = len(cities)
    pop = [Chromosome(dim) for _ in range(pop_size)]
    for ind in pop:
        ind.evaluate(cities)

    best = min(pop, key=lambda ind: ind.fitness)
    print(f'Init best distance: {best.fitness:.2f}')

    for gen in range(1, generations + 1):
        for i, target in enumerate(pop):
            mutant = mutate(pop, i, F)
            trial_genes = crossover(target.genes, mutant, CR)
            trial = Chromosome(dim, genes=trial_genes)
            trial.evaluate(cities)
            if trial.fitness < target.fitness:
                pop[i] = trial
                if trial.fitness < best.fitness:
                    best = trial
        print(f'Gen {gen}: Best = {best.fitness:.2f}')

    return best, best.decode()


if __name__ == '__main__':
    data_file = 'data/TSP1002.txt'
    cities = load_cities(data_file)

    start = time.time()
    best_ind, best_tour = differential_evolution_tsp(
        cities, pop_size=20, F=0.8, CR=0.9, generations=200
    )
    elapsed = time.time() - start
    print(f"Execution Time: {elapsed:.2f} seconds")
    print('Best Distance:', best_ind.fitness)
    print('Best Tour:', best_tour)