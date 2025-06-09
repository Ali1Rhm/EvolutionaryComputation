import csv
import time
import random
import copy
from pathlib import Path

FUNCTIONS = ['+', '-', '*', '/']
TERMINALS = ['x']

def load_dataset(file_path='data/Polynomial.csv'):
    inputs = []
    outputs = []
    path = Path(file_path)
    
    lines = path.read_text(encoding='utf-8').splitlines()
    reader = csv.reader(lines)
    next(reader)

    for row in reader:
        inputs.append(float(row[0]))
        outputs.append(float(row[1]))
    
    return inputs, outputs

class Tree:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.fitness = float('inf')

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_nodes(self):
        nodes = [self]
        if self.left:
            nodes.extend(self.left.get_nodes())
        if self.right:
            nodes.extend(self.right.get_nodes())
        return nodes

    def evaluate(self, x):
        try:
            if self.is_leaf():
                if self.value == 'x':
                    return x
                else:
                    return float(self.value)
            
            left_val = self.left.evaluate(x)
            right_val = self.right.evaluate(x)

            if self.value == '+':
                return left_val + right_val
            elif self.value == '-':
                return left_val - right_val
            elif self.value == '*':
                return left_val * right_val
            elif self.value == '/':
                return left_val / right_val if right_val != 0 else 1.0
        except (ValueError, OverflowError):
            return 1e9

    def __repr__(self):
        if self.is_leaf():
            return str(self.value)
        return f"({self.left.__repr__()} {self.value} {self.right.__repr__()})"

def generate_random_tree(max_depth=4, current_depth=0):
    if current_depth >= max_depth or random.random() < 0.4:
        terminal = random.choice(TERMINALS + [str(random.randint(-5, 5))])
        return Tree(terminal)
    
    func = random.choice(FUNCTIONS)
    left = generate_random_tree(max_depth, current_depth + 1)
    right = generate_random_tree(max_depth, current_depth + 1)
    return Tree(func, left, right)

def get_tree_fitness(tree, inputs, outputs):
    error = 0.0
    for x, y_true in zip(inputs, outputs):
        y_pred = tree.evaluate(x)
        if y_pred is None or abs(y_pred) > 1e9:
            return float('inf')
        error += (y_pred - y_true)**2
    return error / len(inputs) if inputs else float('inf')

def crossover(parent1, parent2):
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    nodes1 = child1.get_nodes()
    node1_to_replace = random.choice(nodes1)
    
    nodes2 = child2.get_nodes()
    node2_subtree = copy.deepcopy(random.choice(nodes2))

    if node1_to_replace == child1:
        return node2_subtree, child2

    parent_of_node1 = next(p for p in nodes1 if p.left == node1_to_replace or p.right == node1_to_replace)

    if parent_of_node1.left == node1_to_replace:
        parent_of_node1.left = node2_subtree
    else:
        parent_of_node1.right = node2_subtree

    return child1, child2

def mutate(tree, max_new_subtree_depth=4):
    mutated_tree = copy.deepcopy(tree)
    nodes = mutated_tree.get_nodes()
    
    node_to_replace = random.choice(nodes)
    
    new_subtree = generate_random_tree(max_depth=max_new_subtree_depth)

    if node_to_replace == mutated_tree:
        return new_subtree

    parent_nodes = [p for p in nodes if not p.is_leaf()]
    for p in parent_nodes:
        if p.left == node_to_replace:
            p.left = new_subtree
            return mutated_tree
        if p.right == node_to_replace:
            p.right = new_subtree
            return mutated_tree
    
    return mutated_tree

ITERATIONS = 20
POPULATION_SIZE = 200
ELITISM_COUNT = 5
TOURNAMENT_SIZE = 7
MUTATION_CHANCE = 0.25
CROSSOVER_CHANCE = 0.7
MUTATION_MAX_DEPTH = 4

inputs, outputs = load_dataset()

start_time = time.time()

population = []
while len(population) < POPULATION_SIZE:
    tree = generate_random_tree()
    tree.fitness = get_tree_fitness(tree, inputs, outputs)
    if tree.fitness != float('inf'):
        population.append(tree)

best_solution_ever = None

for i in range(ITERATIONS):
    population.sort(key=lambda t: t.fitness)

    if best_solution_ever is None or population[0].fitness < best_solution_ever.fitness:
        best_solution_ever = population[0]

    print(f'Iteration {i+1:3} | Best MSE: {population[0].fitness:,.4f} | Best Ever: {best_solution_ever.fitness:,.4f}')

    next_generation = []
    next_generation.extend(population[:ELITISM_COUNT])

    while len(next_generation) < POPULATION_SIZE:
        
        parent1 = random.sample(population, TOURNAMENT_SIZE)
        parent1.sort(key=lambda t: t.fitness)
        parent1 = parent1[0]

        if random.random() < CROSSOVER_CHANCE:
            tournament = random.sample(population, TOURNAMENT_SIZE)
            tournament.sort(key=lambda t: t.fitness)
            parent2 = tournament[0]
            child1, _ = crossover(parent1, parent2)
        else:
            child1 = copy.deepcopy(parent1)

        if random.random() < MUTATION_CHANCE:
            child1 = mutate(child1, max_new_subtree_depth=MUTATION_MAX_DEPTH)
            
        child1.fitness = get_tree_fitness(child1, inputs, outputs)

        if child1.fitness != float('inf'):
            next_generation.append(child1)
    
    population = next_generation

execution_time = time.time() - start_time

print("\n--- Genetic Programming Finished ---")
print(f"Execution Time: {execution_time:.2f} seconds")
print(f"Best solution MSE: {best_solution_ever.fitness:,.4f}")
print(f"Best solution expression: {best_solution_ever}")