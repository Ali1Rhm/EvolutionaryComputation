import random
from pathlib import Path
import time

start_time = time.time()

N = 4
matrix = []
permutation = [2, 3, 1, 4]
permutated_matrix = []

for row in range(N):
    matrix.append([0 for _ in range(N)])
    permutated_matrix.append([0 for _ in range(N)])

path = Path('data/BRP4.txt')
lines = path.read_text(encoding='utf-8').splitlines()
for line in lines:
    indexes = line.split()
    row = int(indexes[0]) - 1
    col = int(indexes[1]) - 1
    matrix[row][col] = 1

for i, row in enumerate(permutated_matrix):
    for j in range(N):
        row[j] = matrix[i][permutation[j]-1]

temp_matrix = permutated_matrix[:]
for i in range(N):
    temp_matrix[i] = permutated_matrix[permutation[i] - 1]
permutated_matrix = temp_matrix

print(f'{time.time() - start_time}')