import random
import math
import copy
import time

import pygame

MAX_DEPTH = 50
POPULATION_SIZE = 100
ITERATIONS = 2000
MATING_POOL_SIZE = 100
MUTATION_RATE = 0.2
POINT_MUTATION_RATE = 0.005
DIVERSITY_INJECTION_RATE = 0.01

N = 16
PUZZLE_GOAL = list(range(1, N+1)) + [-1]
PUZZLE_INIT = [ 12, 1, 10, 2,
                7, 11, 4, 14,
                5, -1, 9, 15,
                8, 13, 6, 3 ]
FUNCTIONS = ['seq', 'if_greater']
TERMINALS = ['up', 'right', 'down', 'left']

TILE_SIZE = 100
MARGIN = 5
GRID_SIZE = 4
WINDOW_SIZE = TILE_SIZE * GRID_SIZE + MARGIN * (GRID_SIZE + 1)

class Tree:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.fitness = float('inf')
        self.result_state = []

    def is_leaf(self):
        return self.left is None and self.right is None

    def evaluate(self):
        moves = self.get_moves(PUZZLE_INIT)
        current = copy.copy(PUZZLE_INIT)
        wrong = 0
        for m in moves:
            new = self.apply_move(current, m)
            if new:
                current = new
            else:
                wrong += 1
        self.result_state = current
        self.fitness = get_tree_fitness(current, len(moves), wrong, self)

    def get_moves(self, state):
        if self.is_leaf():
            return [self.value]
        if self.value == 'seq':
            return self.left.get_moves(state) + self.right.get_moves(state)
        if self.value.startswith('if_greater'):
            a, b = map(int, self.value[self.value.find('(')+1:-1].split(','))
            cond = manhattan_distance(a, state.index(a)) > manhattan_distance(b, state.index(b))
            branch = self.left if cond else self.right
            return branch.get_moves(state)
        return []

    def apply_move(self, state, move):
        if move == 'up':    return move_up(state)
        if move == 'down':  return move_down(state)
        if move == 'left':  return move_left(state)
        if move == 'right': return move_right(state)
        return None


def get_dimension(st):
    n = int(math.sqrt(len(st)))
    if n*n != len(st): raise ValueError
    return n


def find_blank(st):
    return st.index(-1)


def move_up(st):
    if find_blank(st) < get_dimension(st): return None
    new = st.copy(); i = find_blank(st); j = i - get_dimension(st)
    new[i], new[j] = new[j], new[i]
    return new

def move_down(st):
    n = get_dimension(st)
    if find_blank(st) >= n*(n-1): return None
    new = st.copy(); i = find_blank(st); j = i + n
    new[i], new[j] = new[j], new[i]
    return new


def move_left(st):
    if find_blank(st) % get_dimension(st) == 0: return None
    new = st.copy(); i = find_blank(st); j = i - 1
    new[i], new[j] = new[j], new[i]
    return new


def move_right(st):
    n = get_dimension(st)
    if (find_blank(st)+1) % n == 0: return None
    new = st.copy(); i = find_blank(st); j = i + 1
    new[i], new[j] = new[j], new[i]
    return new


def state_manhattan_distance(st):
    dist = 0
    for v in st:
        if v == -1: continue
        idx = st.index(v)
        dist += manhattan_distance(v, idx)
    return dist


def manhattan_distance(val, idx):
    n = int(N**0.5)
    goal_idx = PUZZLE_GOAL.index(val)
    x1, y1 = divmod(idx, n)
    x2, y2 = divmod(goal_idx, n)
    return abs(x1-x2) + abs(y1-y2)


def get_tree_fitness(state, total_moves, wrong_moves, tree):
    distance = state_manhattan_distance(state)
    correct = sum(1 for i, v in enumerate(state) if v == PUZZLE_GOAL[i])
    size_penalty = 0.1 * get_tree_depth(tree)
    score = distance + - 1.5*correct
    return score


def tournament_selection(pop, k=3):
    return min(random.sample(pop, k), key=lambda t: t.fitness)


def get_tree_depth(t):
    if t is None or t.is_leaf(): return 1
    return 1 + max(get_tree_depth(t.left), get_tree_depth(t.right))


def generate_random_tree(depth=3, first=True):
    if not first and (depth<=0 or (depth>1 and random.random()<0.5)):
        return Tree(random.choice(TERMINALS))
    fn = random.choice(FUNCTIONS)
    if fn == 'if_greater':
        a, b = random.sample(range(1, N), 2)
        fn = f"if_greater({a},{b})"
    left = generate_random_tree(depth-1, False)
    right = generate_random_tree(depth-1, False)
    return Tree(fn, left, right)


def get_sub_trees(root, parent=None, is_left=None, res=None):
    if res is None: res = []
    if root and not root.is_leaf():
        res.append((root, parent, is_left))
        get_sub_trees(root.left, root, True, res)
        get_sub_trees(root.right, root, False, res)
    return res


def crossover(p1, p2):
     c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
     subs1 = get_sub_trees(c1)
     subs2 = get_sub_trees(c2)
     if not subs1 or not subs2: return c1, c2
     s1, par1, left1 = random.choice(subs1)
     s2, par2, left2 = random.choice(subs2)
     if par1 is None:
         c1 = s2
     else:
         if left1:
             par1.left = s2
         else:
             par1.right = s2
     if par2 is None:
         c2 = s1
     else:
         if left2:
             par2.left = s1
         else:
             par2.right = s1
     if get_tree_depth(c1) > MAX_DEPTH: c1 = generate_random_tree(MAX_DEPTH)
     if get_tree_depth(c2) > MAX_DEPTH: c2 = generate_random_tree(MAX_DEPTH)
     c1.evaluate(); c2.evaluate()
     return c1, c2


def point_mutation(tree):
    nodes = get_sub_trees(tree)
    if not nodes and tree.is_leaf():
        tree.value = random.choice(TERMINALS)
        tree.evaluate()
        return tree
    node, parent, is_left = random.choice(nodes)
    if node.is_leaf():
        node.value = random.choice(TERMINALS)
    else:
        fn = random.choice(FUNCTIONS)
        if fn == 'if_greater': a, b = random.sample(range(1, N), 2); fn = f"if_greater({a},{b})"
        node.value = fn
    tree.evaluate()
    return tree


def subtree_mutation(tree):
    subs = get_sub_trees(tree)
    if not subs: return tree
    _, par, left = random.choice(subs)
    new_sub = generate_random_tree(random.randint(3, 8))
    if par is None:
        tree = new_sub
    else:
        if left: par.left = new_sub
        else:     par.right = new_sub
    if get_tree_depth(tree) > MAX_DEPTH: tree = generate_random_tree(MAX_DEPTH)
    tree.evaluate()
    return tree


pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("n-Puzzle")
font = pygame.font.SysFont(None, 48)
clock = pygame.time.Clock()
final_state = PUZZLE_INIT

start = time.time()

population = []
for _ in range(POPULATION_SIZE):
    t = generate_random_tree(depth=10)
    t.evaluate()
    population.append(t)

running = True
for it in range(1, ITERATIONS+1):
    if not running:
        break
    mating = [tournament_selection(population) for _ in range(MATING_POOL_SIZE)]
    children = []
    random.shuffle(mating)
    for i in range(0, MATING_POOL_SIZE, 2):
        c1, c2 = crossover(mating[i], mating[i+1])
        children += [c1, c2]

    for i in range(len(children)):
        if random.random() < MUTATION_RATE:
            children[i] = subtree_mutation(children[i])
        elif random.random() < POINT_MUTATION_RATE:
            children[i] = point_mutation(children[i])

    for _ in range(int(POPULATION_SIZE * DIVERSITY_INJECTION_RATE)):
        r = generate_random_tree(depth=10)
        r.evaluate()
        children.append(r)

    population += children
    population = sorted(population, key=lambda x: x.fitness)[:POPULATION_SIZE]

    final_state_fitness = population[0].fitness
    final_state = population[0].result_state
    print(f"Iteration {it} | Best Fitness: {final_state_fitness:.2f} | Result State: {population[0].result_state}")

    screen.fill((255, 255, 255))

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            value = final_state[i * GRID_SIZE + j]
            x = j * TILE_SIZE + (j + 1) * MARGIN
            y = i * TILE_SIZE + (i + 1) * MARGIN
            rect_color = (209, 143, 90)
            if value == -1: rect_color = (0, 0, 0)
            pygame.draw.rect(screen, rect_color, (x, y, TILE_SIZE, TILE_SIZE))
            if value != -1:
                text_surface = font.render(str(value), True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
                screen.blit(text_surface, text_rect)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

print(f"Execution Time: {time.time() - start:.2f} s")