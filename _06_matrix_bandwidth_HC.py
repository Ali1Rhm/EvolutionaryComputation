import random
from pathlib import Path
import time

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

def get_neighbors(permutation, num_neighbors=50):
    neighbors = []

    while len(neighbors) < num_neighbors:
        swaps = []
        for _ in range(10):
            swaps.append((random.randint(0, len(permutation)-1), random.randint(0, len(permutation)-1)))
        neighbor = apply_swaps(permutation.copy(), swaps)
        neighbors.append(neighbor)

    return neighbors

def apply_swaps(perm: list, swaps: list):
    for i, j in swaps:
        perm[i], perm[j] = perm[j], perm[i]
    return perm

def hill_climb(edge_list, max_steps=1000, neighbors_per_step=50, num_points=10, log=True, initial_perm=None):
    points = []
    points_scores = []
    if not initial_perm:
        points = [random_permutation(N) for _ in range(num_points)]
        points_scores = [get_bandwidth_from_edges(point, edge_list) for point in points]
    else:
        num_points = 1
        points.append(initial_perm)
        points_scores.append(get_bandwidth_from_edges(initial_perm, edge_list))
    
    best_score = 0
    best_permutation = None
    for t in range(max_steps):
        for i in range(num_points):
            best_neighbor = points[i]
            best_neighbor_score = points_scores[i]

            for neighbor in get_neighbors(points[i], num_neighbors=neighbors_per_step):
                neighbor_score = get_bandwidth_from_edges(neighbor, edge_list)
                if  neighbor_score < best_neighbor_score:
                    best_neighbor = neighbor
                    best_neighbor_score = neighbor_score
                    points[i] = best_neighbor
                    points_scores[i] = best_neighbor_score

        best_score = min(points_scores)
        best_permutation = points[points_scores.index(best_score)]
        if log:
            print(f"HC Iteration {t:3d} | Best Bandwidth: {best_score}")
    
    return best_permutation, best_score

if __name__ == '__main__':
    N = 1624
    edge_list = get_edge_list(Path('data/BRP1624.txt'), N)
    print(f"Original matrix bandwidth: {get_bandwidth_from_edges(list(range(N)), edge_list)}")

    start_time = time.time()
    hill_climb(edge_list)
    print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")