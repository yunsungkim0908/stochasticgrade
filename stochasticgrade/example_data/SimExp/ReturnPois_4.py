import random 

def simulate_bernoulli(p = 0.4):
    if random.uniform(0, 1) < p: 
        return 1 
    return 0 

def simulate_binomial(n = 20, p = 0.4): 
    x = 0 
    for i in range(n): 
        x = x + simulate_bernoulli(p) 
    return x

def simulate_exponential(): 
    LAMBDA = 3.1
    n = 1000
    x = simulate_binomial(n, (1 / LAMBDA) / n) 
    return x