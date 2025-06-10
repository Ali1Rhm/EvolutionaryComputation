import random
from pathlib import Path
import time

from _06_matrix_bandwidth_HC import hill_climb

def get_edge_list(path: Path, N: int):
    edges = []
    lines = path.read_text(encoding='utf-8').splitlines()
    for line in lines:
        u_str, v_str = line.split()
        u, v = int(u_str)-1, int(v_str)-1
        if u < v:
            edges.append((u, v))
        else:
            edges.append((v, u))
    return edges

def get_bandwidth_from_edges(permutation: list, edges: list):  
    pos = [0] * len(permutation)
    for idx, node in enumerate(permutation):
        pos[node] = idx
    
    max_bw = 0
    for u, v in edges:
        d = abs(pos[u] - pos[v])
        if d > max_bw:
            max_bw = d
    return max_bw

def random_permutation(N: int):
    perm = list(range(N))
    random.shuffle(perm)
    return perm

def diff_sequence(current: list, target: list) -> list:
    N = len(current)
    cur = current.copy()
    pos = {node: idx for idx, node in enumerate(cur)}
    swaps = []

    for i in range(N):
        if cur[i] != target[i]:
            j = pos[target[i]]
            swaps.append((i, j))
            cur[i], cur[j] = cur[j], cur[i]
            pos[cur[j]] = j
            pos[cur[i]] = i

    return swaps

def combine(old_velocity: list,
            diff_to_pbest: list,
            diff_to_gbest: list,
            w: float = 0.6,
            c1: float = 0.4,
            c2: float = 0.4,
            max_velocity_len: int = 50) -> list:
    keep = max(1, int(w * len(old_velocity)))
    new_vel = old_velocity[:keep]

    for s in diff_to_pbest:
        if random.random() < c1:
            new_vel.append(s)

    for s in diff_to_gbest:
        if random.random() < c2:
            new_vel.append(s)

    seen = set()
    unique_vel = []
    for s in new_vel:
        if s not in seen:
            unique_vel.append(s)
            seen.add(s)
    new_vel = unique_vel

    if not new_vel:
        pool = diff_to_pbest + diff_to_gbest
        if pool:
            new_vel.append(random.choice(pool))
        else:
            idx = random.randint(0, N - 2)
            new_vel.append((idx, idx + 1))

    if len(new_vel) > max_velocity_len:
        new_vel = random.sample(new_vel, max_velocity_len)

    random.shuffle(new_vel)
    return new_vel

def apply_swaps(perm: list, swaps: list):
    for i, j in swaps:
        perm[i], perm[j] = perm[j], perm[i]
    return perm

N = 1624
swarm_size = 50
iterations = 600

edge_list = get_edge_list(Path('data/BRP1624.txt'), N)
print(f"Original matrix bandwidth: {get_bandwidth_from_edges(list(range(N)), edge_list)}")

swarm = [random_permutation(N) for _ in range(swarm_size)]
velocities = [[] for _ in swarm]
p_bests = [p.copy() for p in swarm]
p_best_scores = [get_bandwidth_from_edges(p, edge_list) for p in p_bests]
g_best = p_bests[p_best_scores.index(min(p_best_scores))]
g_best_score = min(p_best_scores)

start_time = time.time()

for t in range(1, iterations+1):
    w = 0.7 - 0.2 * (t / iterations)
    c1 = 0.3 * (1 - t / iterations)
    c2 = 0.3 * (1 - t / iterations)

    for i in range(swarm_size):
        curr = swarm[i]
        v_old = velocities[i]
        diff_p = diff_sequence(curr, p_bests[i])
        diff_g = diff_sequence(curr, g_best)
        v_new = combine(v_old, diff_p, diff_g, w, c1, c2)
        velocities[i] = v_new
        swarm[i] = apply_swaps(curr.copy(), v_new)
        score = get_bandwidth_from_edges(swarm[i], edge_list)
        if random.random() < 0.3:
            refined, refined_score = hill_climb(edge_list, max_steps=3, neighbors_per_step=10, initial_perm=swarm[i], log=False)
            if refined_score < score:
                swarm[i] = refined
                score = refined_score

        if score < p_best_scores[i]:
            p_bests[i] = swarm[i].copy()
            p_best_scores[i] = score

        if score < g_best_score:
            g_best = swarm[i].copy()
            g_best_score = score

    polished, polished_score = hill_climb(edge_list, max_steps=5, neighbors_per_step=20, initial_perm=g_best, log=False)
    if polished_score < g_best_score:
      g_best, g_best_score = polished, polished_score

    print(f"Iteration {t:3d} | Best Bandwidth: {g_best_score}")


print(f"Final Best Bandwidth: {g_best_score}")
print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")