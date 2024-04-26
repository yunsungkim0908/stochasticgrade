import argparse
import configparser
import json
import numpy as np
import os
import shutil

from stochasticgrade.constants import DATA_DIR
from stochasticgrade.grade import StochasticGrade
from stochasticgrade.score import *
        

def choose_n(qid, false_acceptance_rate, false_rejection_rate, n_soln_samples,  
             scorer, dtype, func_name, min_n, max_n, M=1000):
    """
    Determine the sample size that satisfies the provided FAR when run on
    the closest error program. Out of `M` identical error programs, only a
    `false_acceptance_rate` fraction of them will be falsely accepted.
    
    qid (str):                     the question ID
    false_acceptance_rate (float): the false acceptance rate for the error program
    false_rejection_rate (float):  the false rejection rate for the correct programs
    n_soln_samples (int):          the number of solution samples
    scorer (Scorer):               the scoring function to use
    dtype (str):                   the data type of the problem
    func_name (str):               the name of the function from which samples are collected
    min_n (int):                   the minimum number of samples to test
    max_n (int):                   the maximum number of samples to test
    M (int):                       the number of times to execute the grading algorithm
    
    Returns: 
    best_n (int): the number of samples that satisfies the FAR, or max_n if no value is found
    """
    # TODO: Change M back to 1000
    
    print('\n- - - - - CHOOSE N - - - - -\n')
    # Determine the sample sizes to evaluate, in powers of 2
    sizes = []
    curr = min_n
    while curr < max_n:
        sizes.append(curr)
        curr *= 2
    sizes.append(max_n)
    
    # Create the IDs for the closest error program
    sids = [f'closest_error_{i+1}' for i in range(M)]
    for sid in sids:
        sid_dir = os.path.join(DATA_DIR, qid, 'setup', sid)
        if not os.path.isdir(sid_dir):
            os.makedirs(sid_dir)
        src = os.path.join(DATA_DIR, qid, 'setup', 'closest_error', 'program.py')
        dst = os.path.join(sid_dir, 'program.py')
        shutil.copy(src, dst)
    
    # Execute the grading algorithm for each sample size until the FAR is satisfied
    found = False
    for size in sizes:
        print(f'Testing size {size}')
        best_n = size
        algorithm = StochasticGrade(qid, scorer, [size], false_rejection_rate, dtype, func_name,
                                    n_soln_samples=n_soln_samples)
        # Run the grading algorithm on the error program M times
        false_acceptances = 0    
        results = algorithm.grade_parallel(sids)
        for sid in results:
            false_acceptances += 1 if results[sid][0] else 0
        calculated_far = false_acceptances / M
        
        # Determine if the calculated FAR is sufficient or if we need more samples
        if calculated_far <= false_acceptance_rate:
            print(f'Calculated FAR: {calculated_far} is within the tolerance of {false_acceptance_rate}.\n')
            print(f'The best value of N: {best_n}\n')
            found = True
            break
        print(f'Calculated FAR: {calculated_far} is beyond the tolerance of {false_acceptance_rate}.')
    
    if not found:
        print(f'\nNo best_n found within the sample size maximum. Returning max_n of {max_n}.\n')
    
    print('Cleaning up directories.')
    for sid in sids:
        sid_dir = os.path.join(DATA_DIR, qid, 'setup', sid)
        shutil.rmtree(sid_dir)
    print('Success!\n\n')
    
    return best_n


if __name__ == '__main__':
    
    # Load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('qid', type=str, help='the question ID')
    parser.add_argument('--far', type=float, default=0.1, help='the false acceptance rate')
    parser.add_argument('--M', type=int, default=1000, 
                        help='the number of times to execute the grading algorithm to check FAR')
    args = parser.parse_args()
    
    qid = args.qid
    far = args.far
    M = args.M
    
    # Load config file
    config = configparser.ConfigParser()
    if os.path.isfile(os.path.join(DATA_DIR, qid, 'setup', 'config.ini')):   
        config.read(os.path.join(DATA_DIR, qid, 'setup', 'config.ini'))
       
    # Load model parameters
    dtype = config['Parameters']['dtype']
    n_soln_samples = int(config['Parameters']['n_soln_samples'])
    min_n = int(config['Parameters']['min_n'])
    max_n = int(config['Parameters']['max_n'])
    scorer_name = config['Parameters']['scorer']
    scorer_map = make_scorer_map()
    scorer = scorer_map[scorer_name]
    func_name = config['Parameters']['func_name']
    frr = float(config['Parameters']['frr'])
               
    # Run choose_n and save the best selection for the number of samples that obey the FAR
    best_n = choose_n(
        qid, far, frr, n_soln_samples, scorer, dtype, func_name, min_n, max_n, M=M
    )
    with open(os.path.join(DATA_DIR, qid, 'setup', f'best_n_far={far}.txt'), 'w') as f:
        f.write(f'best_n={best_n}')
    
                
            
            
