import random
import math
import copy
import time

import pygame

MAX_DEPTH = 100
POPULATION_SIZE = 100
MATING_POOL_SIZE = 100
ITERATIONS = 2000
MUTATION_RATE = 0.2
POINT_MUTATION_RATE = 0.005
DIVERSITY_INJECTION_RATE = 0.01

# N اصلی را منهای یک نمایید
N = 8

# حالت هدف را اینجا مشخص کنید
PUZZLE_GOAL = list(range(1, N+1)) + [-1]

# حالت اولیه را اینجا مشخص کنید (حتما باید قابل حل باشد)
PUZZLE_INIT = [8, 6, 7,
               2, 5, 4,
               3, 1, -1]

FUNCTIONS = ['seq', 'if_greater']
TERMINALS = ['up', 'right', 'down', 'left']

TILE_SIZE = 100
MARGIN = 5
GRID_SIZE = 3
WINDOW_SIZE = TILE_SIZE * GRID_SIZE + MARGIN * (GRID_SIZE + 1)
ANIMATION_DELAY = 200

def get_dimension(st):
    return int(math.sqrt(len(st)))

def find_blank(st):
    return st.index(-1)

def move_up(st):
    n = get_dimension(st)
    i = find_blank(st)
    if i < n: return None
    new = st.copy()
    j = i - n
    new[i], new[j] = new[j], new[i]
    return new

def move_down(st):
    n = get_dimension(st)
    i = find_blank(st)
    if i >= n*(n-1): return None
    new = st.copy()
    j = i + n
    new[i], new[j] = new[j], new[i]
    return new

def move_left(st):
    n = get_dimension(st)
    i = find_blank(st)
    if i % n == 0: return None
    new = st.copy()
    j = i - 1
    new[i], new[j] = new[j], new[i]
    return new

def move_right(st):
    n = get_dimension(st)
    i = find_blank(st)
    if (i + 1) % n == 0: return None
    new = st.copy()
    j = i + 1
    new[i], new[j] = new[j], new[i]
    return new

def state_manhattan_distance(st):
    total = 0
    for idx, v in enumerate(st):
        if v == -1: continue
        total += manhattan_distance(v, idx)
    return total

def manhattan_distance(val, idx):
    n = GRID_SIZE
    goal_idx = PUZZLE_GOAL.index(val)
    x1, y1 = divmod(idx, n)
    x2, y2 = divmod(goal_idx, n)
    return abs(x1 - x2) + abs(y1 - y2)

class Tree:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.fitness = float('inf')
        self.result_state = []
        self.moves = []

    def is_leaf(self):
        return self.left is None and self.right is None

    def evaluate(self):
        self.moves = self.get_moves(PUZZLE_INIT)
        current = PUZZLE_INIT.copy()
        wrong = 0
        for m in self.moves:
            new = self.apply_move(current, m)
            if new:
                current = new
            else:
                wrong += 1
        self.result_state = current
        self.fitness = get_tree_fitness(current, len(self.moves), wrong, self)

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

def get_tree_depth(t):
    if t is None or t.is_leaf(): return 1
    return 1 + max(get_tree_depth(t.left), get_tree_depth(t.right))

def get_tree_fitness(state, total_moves, wrong_moves, tree):
    distance = state_manhattan_distance(state)
    correct = sum(1 for i, v in enumerate(state) if v == PUZZLE_GOAL[i])
    size_penalty = 0.1 * get_tree_depth(tree)
    return distance - 1.5 * correct + size_penalty + wrong_moves * 2

def tournament_selection(pop, k=3):
    return min(random.sample(pop, k), key=lambda t: t.fitness)

def generate_random_tree(depth=3, first=True):
    if not first and (depth <= 0 or (depth > 1 and random.random() < 0.5)):
        return Tree(random.choice(TERMINALS))
    fn = random.choice(FUNCTIONS)
    if fn == 'if_greater':
        a, b = random.sample(range(1, N), 2)
        fn = f"if_greater({a},{b})"
    left = generate_random_tree(depth - 1, False)
    right = generate_random_tree(depth - 1, False)
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
    if not subs1 or not subs2:
        c1.evaluate(); c2.evaluate()
        return c1, c2
    s1, par1, left1 = random.choice(subs1)
    s2, par2, left2 = random.choice(subs2)
    if par1 is None:
        c1 = copy.deepcopy(s2)
    elif left1:
        par1.left = copy.deepcopy(s2)
    else:
        par1.right = copy.deepcopy(s2)
    if par2 is None:
        c2 = copy.deepcopy(s1)
    elif left2:
        par2.left = copy.deepcopy(s1)
    else:
        par2.right = copy.deepcopy(s1)
    if get_tree_depth(c1) > MAX_DEPTH: c1 = generate_random_tree(MAX_DEPTH)
    if get_tree_depth(c2) > MAX_DEPTH: c2 = generate_random_tree(MAX_DEPTH)
    c1.evaluate(); c2.evaluate()
    return c1, c2

def point_mutation(tree):
    nodes = get_sub_trees(tree)
    if tree.is_leaf():
        tree.value = random.choice(TERMINALS)
    else:
        node, _, _ = random.choice(nodes)
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
    node, par, left = random.choice(subs)
    new_sub = generate_random_tree(random.randint(3, 8))
    if par is None:
        tree = new_sub
    elif left:
        par.left = new_sub
    else:
        par.right = new_sub
    if get_tree_depth(tree) > MAX_DEPTH: tree = generate_random_tree(MAX_DEPTH)
    tree.evaluate()
    return tree

if __name__ == '__main__':
    start_time = time.time()
    population = [generate_random_tree(depth=10) for _ in range(POPULATION_SIZE)]
    for t in population:
        t.evaluate()

    for it in range(1, ITERATIONS+1):
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
        population = sorted(population, key=lambda t: t.fitness)[:POPULATION_SIZE]
        if it % 100 == 0 or it == 1:
            best = population[0]
            print(f"Iteration {it} | Best Fitness: {best.fitness:.2f} | Moves: {len(best.moves)}")

    best_tree = population[0]
    best_moves = best_tree.moves
    print(f"Execution Time: {time.time() - start_time:.2f}")
    print(f"Best Fitness: {best_tree.fitness:.2f}")
    print("Best Move Sequence:", best_moves)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("n-Puzzle Solution")
    font = pygame.font.SysFont(None, 48)
    clock = pygame.time.Clock()

    current = PUZZLE_INIT.copy()
    move_index = 0
    running = True
    last_move_time = pygame.time.get_ticks()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        now = pygame.time.get_ticks()
        if move_index < len(best_moves) and now - last_move_time >= ANIMATION_DELAY:
            m = best_moves[move_index]
            new = Tree(None).apply_move(current, m)
            if new: current = new
            move_index += 1
            last_move_time = now

        screen.fill((255, 255, 255))
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                v = current[i*GRID_SIZE + j]
                x = j * TILE_SIZE + (j+1)*MARGIN
                y = i * TILE_SIZE + (i+1)*MARGIN
                color = (209, 143, 90) if v != -1 else (0,0,0)
                pygame.draw.rect(screen, color, (x,y,TILE_SIZE,TILE_SIZE))
                if v != -1:
                    txt = font.render(str(v), True, (255,255,255))
                    rect = txt.get_rect(center=(x+TILE_SIZE//2, y+TILE_SIZE//2))
                    screen.blit(txt, rect)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()