import json
import numpy as np
import os

from stochasticgrade.constants import DATA_DIR
from stochasticgrade.grade import StochasticGrade

def initialize(file_path, qid):
    """
    Initialize the error directory and the FAR for running the script.
    
    file_path (str): the path to the closest error program, as a .py file
    qid (str):       the question ID
    """

    print('\n- - - - - INITIALIZATION - - - - -\n')
    
    # Create the directory for the closest error program
    print('Creating the directory for the closest error program.')
    if not os.isdir(os.path.join(DATA_DIR, qid, 'closest_error', 'closest_error')):
        os.makedirs(os.path.join(DATA_DIR, qid, 'closest_error', 'closest_error'))
        
    # Move the read program into the directory using the sid `closest_error`
    if not file_path.endswith('.py'):
        raise ValueError('Please provide the path to a valid .py file for the closest error program.')
    with open(file_path) as f:
        error_program = f.read()
    with open(os.path.join(DATA_DIR, qid, 'closest_error', 'closest_error', 'program.py'), 'w') as f:
        f.write(error_program)
    
    # Request a FAR to add to the config file
    print('\nPlease select the desirable false acceptance rate for the closest error program.')
    print('This is a float, e.g. 0.1.')
    far = input('FAR: ')
     
    # Update the config file
    config = configparser.ConfigParser()
    if os.path.isfile(os.path.join(DATA_DIR, qid, 'config.ini')):   
        config.read(os.path.join(DATA_DIR, qid, 'config.ini'))
    config['Parameters']['far'] = far
    with open(os.path.join(DATA_DIR, qid, 'config.ini'), 'w') as f:
        config.write(f)
        

def choose_n(qid, false_acceptance_rate, false_rejection_rate, 
             scorer, dtype, func_name, min_n=200, max_n=500000, M=10, test_label='', test_args=[]):
    """
    Determine the sample size that satisfies the provided FAR when run on
    the closest error program. Out of `M` identical error programs, only a
    `false_acceptance_rate` fraction of them will be falsely accepted.
    
    qid (str):                     the question ID
    false_acceptance_rate (float): the false acceptance rate for the error program
    false_rejection_rate (float):  the false rejection rate for the correct programs
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
    # Determine the sample sizes to evaluate, in powers of 2
    sizes = []
    curr = min_n
    while curr < max_n:
        sizes.append(curr)
        curr *= 2
    sizes.append(max_n)
    
    # Number of solution samples used
    soln_sample_path = os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 'samples.npy')
    if os.path.isfile(soln_sample_path):
        soln_samples = np.load(soln_sample_path, allow_pickle=True)
    else:
        raise ValueError('No solution samples found. Be sure to execute preprocess.py!')
    num_soln_samples = len(soln_samples)
    
    # The ID for the closest error program
    sid = 'closest_error'
    
    # Execute the grading algorithm for each sample size until the FAR is satisfied
    for size in sizes:
        print(f'Testing size {size}')
        best_n = size
        algorithm = StochasticGrade(qid, scorer, [size], false_rejection_rate, dtype, func_name,
                                    num_soln_samples=num_soln_samples, test_label=test_label, test_args=test_args)
        # Run the grading algorithm on the error program M times
        false_acceptances = 0
        for _ in range(M):
            result = algorithm.grade(sid)
            accepted = result[0]
            if accepted: 
                false_acceptances += 1
        simulated_far = false_acceptances / M
        
        # Determine if the calculated FAR is sufficient or if we need more samples
        if simulated_far <= false_acceptance_rate:
            print(f'best_n: {best_n}')
            return best_n
    print(f'No best_n found within the sample size maximum. Returning max_n of {max_n}.')
    return best_n


if __name__ == '__main__':
    
    # Load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('qid', type=str)
    parser.add_argument('file_path', type=str)
    args = parser.parse_args()
    
    qid = args.qid
    file_path = args.file_path
    
    # Initialize the error program directory and FAR
    initialize(file_path)
    
    # Load config file
    config = configparser.ConfigParser()
    if os.path.isfile(os.path.join(DATA_DIR, qid, 'config.ini')):   
        config.read(os.path.join(DATA_DIR, qid, 'config.ini'))
       
    # Load model parameters
    dtype = config['Parameters']['dtype']
    num_soln_samples = int(config['Parameters']['num_soln_samples'])
    min_n = int(config['Parameters']['min_n'])
    max_n = int(config['Parameters']['max_n'])
    scorer_name = config['Parameters']['scorer']
    scorer_map = make_scorer_map()
    scorer = scorer_map[scorer_name]
    func_name = config['Parameters']['func_name']
    far = config['Parameters']['far']
    frr = config['Parameters']['frr']
    
    # Run choose_n and save the best selection for the number of samples that obey the FAR
    best_n = choose_n(
        qid, far, frr, scorer, dtype, func_name, min_n=min_n, max_n=max_n
    )
    with open(os.path.join(DATA_DIR, qid, 'closest_error', f'best_n_far={far}.txt')) as f:
        f.write(f'best_n = {best_n}')
    
                
            
            
