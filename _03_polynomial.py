import math
from pathlib import Path
import csv
import time
import random
import copy

def load_dataset():
    path = Path('data/Polynomial.csv')
    lines = path.read_text(encoding='utf-8').splitlines()
    reader = csv.reader(lines)
    next(reader)

    inputs = []
    outputs = []

    for row in reader:
        inputs.append(float(row[0]))
        outputs.append(float(row[1]))
    
    return inputs, outputs

class Tree:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.fitness = 0
        self.selection_probability = 0

    def is_leaf(self):
        return self.left is None and self.right is None

    def evaluate(self, x):
        try:
            if self.value == 'x':
                return x
            elif self.value == 'sinx':
                return math.sin(x)
            elif self.value == 'cosx':
                return math.cos(x)
            elif self.value in FUNCTIONS:
                left_val = self.left.evaluate(x)
                right_val = self.right.evaluate(x)

                if self.value == '+':
                    return left_val + right_val
                elif self.value == '-':
                    return left_val - right_val
                elif self.value == '*':
                    return left_val * right_val
                elif self.value == '/':
                    return left_val / right_val if right_val != 0 else 1
                elif self.value == '^':
                    if left_val < 0 and not float(right_val).is_integer():
                        return 1
                    return left_val ** right_val
            else:
                return float(self.value)
        except Exception:
            return 1e6

    def __repr__(self):
        if self.is_leaf():
            return str(self.value)
        return f"({self.left} {self.value} {self.right})"

def generate_random_tree(depth=3):
    if depth == 0 or (depth > 1 and random.random() < 0.5):
        if random.random() < 0.5:
            return Tree('x')
        else:
            return Tree(str(random.randint(0, 9)))
    else:
        func = random.choice(FUNCTIONS)
        left = generate_random_tree(depth - 1)
        right = generate_random_tree(depth - 1)
        return Tree(func, left, right)

def get_tree_fitness(tree, inputs, outputs):
    error = 0
    for i, x in enumerate(inputs):
        try:
            y_pred = tree.evaluate(x)
            y_true = outputs[i]
            error += (y_pred - y_true) ** 2
        except Exception:
            error += 1e6
    return error

def get_tree_selection_probability(rank, population_size, s=2):
    return ((2 - s) / population_size) + ((2 * rank * (s - 1)) / (population_size * (population_size - 1)))

def get_sub_trees(tree, parent=None, is_left=None, result=None):
    if result is None:
        result = []

    if tree is not None and not tree.is_leaf():
        result.append((tree, parent, is_left))
        get_sub_trees(tree.left, tree, True, result)
        get_sub_trees(tree.right, tree, False, result)

    return result

def get_tree_depth(tree):
    if tree is None or tree.is_leaf():
        return 1
    return 1 + max(get_tree_depth(tree.left), get_tree_depth(tree.right))

def crossover(first_parent: Tree, second_parent: Tree):
    first_child = copy.deepcopy(first_parent)
    second_child = copy.deepcopy(second_parent)

    first_child_sub_trees = get_sub_trees(first_child)
    second_child_sub_trees = get_sub_trees(second_child)

    if not first_child_sub_trees or not second_child_sub_trees:
        return first_child, second_child

    first_subtree, first_parent_node, is_left1 = random.choice(first_child_sub_trees)
    second_subtree, second_parent_node, is_left2 = random.choice(second_child_sub_trees)

    MIN_DEPTH = 2
    if first_parent_node is None and get_tree_depth(second_subtree) < MIN_DEPTH:
        return first_child, second_child
    if second_parent_node is None and get_tree_depth(first_subtree) < MIN_DEPTH:
        return first_child, second_child

    if first_parent_node is None:
        first_child = second_subtree
    else:
        if is_left1:
            first_parent_node.left = second_subtree
        else:
            first_parent_node.right = second_subtree

    if second_parent_node is None:
        second_child = first_subtree
    else:
        if is_left2:
            second_parent_node.left = first_subtree
        else:
            second_parent_node.right = first_subtree

    first_child.fitness = get_tree_fitness(first_child, inputs, outputs)
    second_child.fitness = get_tree_fitness(second_child, inputs, outputs)

    return first_child, second_child

def mutate(tree: Tree):
    subtrees = get_sub_trees(tree)

    if not subtrees:
        return tree

    subtree_to_replace, parent, is_left = random.choice(subtrees)

    new_subtree = generate_random_tree(random.randint(3, 6))

    if parent is None:
        new_subtree.fitness = get_tree_fitness(new_subtree, inputs, outputs)
        return new_subtree
    else:
        if is_left:
            parent.left = new_subtree
        else:
            parent.right = new_subtree
        
        tree.fitness = get_tree_fitness(tree, inputs, outputs)
        return tree

FUNCTIONS = ['+', '-', '*', '/', '^']
TERMINALS = ['x']
inputs, outputs = load_dataset()
iterations = 20
population_size = 100
mating_pool_size = 100
population = []
mating_pool = []

start_time = time.time()

current_iteration = 1
while current_iteration <= iterations:
    # Initialize population
    if current_iteration == 1:
        while len(population) < population_size:
            tree = generate_random_tree()
            tree.fitness = get_tree_fitness(tree, inputs, outputs)
            population.append(tree)
    
    # Rank individuals based on their fitness and calculate cumulative probabilities
    ranked_population = sorted(population, reverse=True, key=lambda c: c.fitness)
    cumulative_probabilities = []
    cumulative_sum = 0
    for i, tree in enumerate(ranked_population):
        prob = get_tree_selection_probability(i, population_size, s=2)
        tree.selection_probability = prob
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
        offsprings = crossover(mating_pool[cross_over_indexes[0]], mating_pool[cross_over_indexes[1]])
        mating_pool.append(offsprings[0])
        mating_pool.append(offsprings[1])
        i += 2

    # Apply mutations over individual parents
    for i in range(len(mating_pool)):
        mating_pool[i] = mutate(mating_pool[i])

    population_mating_pool = (population + mating_pool)
    population_mating_pool = sorted(population_mating_pool, reverse=True, key=lambda c: c.fitness)
    population = population_mating_pool[-population_size:]

    print(f'Iteration {current_iteration:}')
    print(f'Best tree fitness: {population[-1].fitness:_}\n')

    current_iteration += 1

print(f"\nExecution Time: {time.time() - start_time:.2f} seconds")