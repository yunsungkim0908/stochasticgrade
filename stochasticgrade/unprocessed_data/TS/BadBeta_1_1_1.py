from scipy import stats
import random

def thompson_sampling(history):
    A_sample = stats.beta.rvs(1, 1)
    B_sample = stats.beta.rvs(1, 1)
    if A_sample > B_sample:
        return 'A'
    else:
        return 'B'
    
def main(history):
    res = thompson_sampling(history)
    return 0 if res == 'A' else 1
    