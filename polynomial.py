from pathlib import Path
import csv
import time
import random

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

inputs, outputs = load_dataset()
iterations = 10
population_size = 10
mating_pool_size = 10
population = []
mating_pool = []

start_time = time.time()

print(f"\nExecution Time: {time.time() - start_time:.2f} seconds")