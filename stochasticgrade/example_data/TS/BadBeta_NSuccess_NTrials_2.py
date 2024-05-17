from scipy import stats
import random

def thompson_sampling(history):
    A_a = history['A']['n_success'] + 1
    A_b = history['A']['n_trials'] + 1
    B_a = history['B']['n_success'] + 1
    B_b = history['B']['n_trials'] + 1
    A_sample = stats.beta.rvs(A_a, A_b)
    B_sample = stats.beta.rvs(B_a, B_b)
    if A_sample > B_sample:
        return 'A'
    else:
        return 'B'
    
def main(history):
    res = thompson_sampling(history)
    return 0 if res == 'A' else 1