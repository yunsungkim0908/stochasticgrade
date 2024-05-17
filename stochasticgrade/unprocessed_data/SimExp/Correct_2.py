import random

def simulate_exponential():
    LAMBDA = 3.1
    n = 1000
    time_step = 1 / n
    time_elapsed = 0
    while random.uniform(0, 1) > LAMBDA / n:
        time_elapsed += time_step
    return time_elapsed