"""
preprocess.py
-------------
This file handles loading and parsing data.
It creates the directory structure and components necessary to store and handle data
for a given problem.
It also performs preliminary sampling for the solution program and generates Monte Carlo samples.
"""

import argparse
import configparser
import json
import numpy as np
import os
import shutil

from tqdm import tqdm

from stochasticgrade.constants import DATA_DIR
from stochasticgrade.sample import sample_sid_single, sample_sid_multi, monte_carlo_sample_multi
from stochasticgrade.score import *


def load_qid_data(file_path, qid):
    """
    Load in the student and solution data and create the appropriate
    directories within the corresponding `qid` directory.
    
    Parameters:
    file_path (str):  the path to a directory containing:
                      - solution.py: the solution program
                      - one of the following formats for student data:
                          - [sid].py files directly within the `file_path` directory
                          - a .json file containing sid (str): program (str) mappings
                          
                      OPTIONAL:
                      - closest_error.py: the closest acceptable error program to the solution
                        (this is used if trying to adjust the false acceptance rate)
                      - grading_arguments.json: a .json file containing the arguments needed to run
                        the function to be graded. It is a mapping between variable names and the 
                        argument. The variables must be in the same order.
                      
    qid (str):        the question ID
    
    Returns: 
    None
    """
    
    print('\n\n- - - - - DATA PROCESSING - - - - -\n')
    
    # Create the question ID directory, if it doesn't exist
    print('Creating the question ID directory.')
    if not os.path.isdir(os.path.join(DATA_DIR, qid)):
        os.makedirs(os.path.join(DATA_DIR, qid))
        
    # Create results directory
    print('Creating results directory.')
    if not os.path.isdir(os.path.join(DATA_DIR, qid, 'results')):
        os.makedirs(os.path.join(DATA_DIR, qid, 'results'))
        
    # Create setup directory
    print('Creating setup directory.')
    if not os.path.isdir(os.path.join(DATA_DIR, qid, 'setup')):
        os.makedirs(os.path.join(DATA_DIR, qid, 'setup'))
    
    # Check the validity of the provided file path
    if not os.path.exists(file_path):
        raise ValueError(f'The file path "{file_path}" does not exist. Please enter a valid file path.')
    if not os.path.isdir(file_path):
        raise ValueError(f'Please ensure the file path "{file_path}" is a directory.')
        
    # Load grading arguments
    if 'grading_arguments.json' in os.listdir(file_path):
        print('Loading in the grading arguments.')
        src = os.path.join(file_path, 'grading_arguments.json')
        dst = os.path.join(DATA_DIR, qid, 'setup', 'grading_arguments.json')
        shutil.copy(src, dst)      
    
    # Load solution program data
    print('Loading in the solution program data.')
    if 'solution.py' not in os.listdir(file_path):
        raise ValueError(f'Please create solution program "solution.py" in file path "{file_path}".')
    with open(os.path.join(file_path, 'solution.py')) as f:
        solution_program = f.read()
    
    # Load student program data
    print('Loading in the student program data.')
    student_programs = {}     
    for item in os.listdir(file_path): 
        if item in ['solution.py', 'closest_error.py', 'grading_arguments.json']:
            continue
        if item.endswith('.json'):  # Student data formatted as a .json file
            data = json.load(open(os.path.join(file_path, item)))
            student_programs = {sid: data[sid] for sid in data}
        if item.endswith('.py'):  # Student data formatted as [sid].py files
            with open(os.path.join(file_path, item)) as f:
                student_programs[item[:-3]] = f.read()
    if not student_programs:
        raise ValueError(f'No student programs were found.')
        
    # Create the solution directory with its program
    soln_dir = os.path.join(DATA_DIR, qid, 'solution', 'solution')
    if not os.path.isdir(soln_dir):
        os.makedirs(soln_dir)
    with open(os.path.join(soln_dir, 'program.py'), 'w') as f:
        f.write(solution_program)
                  
    # Create the closest error directory with its program
    if 'closest_error.py' in os.listdir(file_path):
        print('Loading in the closest error program.')
        if not os.path.isdir(os.path.join(DATA_DIR, qid, 'setup', 'closest_error')):
            os.makedirs(os.path.join(DATA_DIR, qid, 'setup', 'closest_error'))
        with open(os.path.join(file_path, 'closest_error.py')) as f:
            error_program = f.read()
        with open(os.path.join(DATA_DIR, qid, 'setup', 'closest_error', 'program.py'), 'w') as f:
            f.write(error_program)
    
    # Create the student ID directories with their programs
    for sid in student_programs: 
        sid_dir = os.path.join(DATA_DIR, qid, 'students', sid)
        if not os.path.isdir(sid_dir):
            os.makedirs(sid_dir)
        with open(os.path.join(sid_dir, 'program.py'), 'w') as f:   
            f.write(student_programs[sid])
        
    print('Success!\n\n')
    
    
def sample_solution(qid, n_samples, dtype, func_name, test_label='', test_args=[]):
    """
    Sample the solution program. In total, `n_samples` samples are generated.
    
    Parameters:
    qid (str):        the question ID
    n_samples (int):  the number of samples to generate
    dtype (str):      the data type of the sample solutions
    func_name (str):  the name of the function to be evaluated
    test_label (str): name of the test case to be evaluated
    test_args (list): arguments to be used for the test case
    
    Returns:
    None
    """
    
    print('- - - - - SOLUTION SAMPLE GENERATION - - - - -\n')
    
    # Generate the solution samples
    print(f'Generating {n_samples} samples from the solution program.')
    sid = 'solution'
    sample_sid_single(sid, qid, n_samples, dtype, func_name, test_label=test_label, test_args=test_args)  
    sample_path = os.path.join(DATA_DIR, qid, 'solution', sid, test_label, 'samples.npy')
    samples = np.load(sample_path, allow_pickle=True)
    if len(samples) == 0:
        os.remove(sample_path)
        raise Exception('Did not successfully generate samples. Did you correctly specify all model parameters?')
    else:
        print(f'Succesfully generated {len(samples)} solution samples!\n\n')
    
    
def monte_carlo(min_n, max_n, qid, dtype, func_name, scorer, M=1000, max_parallel=20, 
                save_samples=False, test_label='', test_args=[], proj_method=''):
    """
    Sample `M` sample sets of size `max_n` from the solution program.
    Calculate the scores under each scorer for determining the FRR when grading.
    Scores are calculated in powers of 2 from `min_n` to `max_n`.
    
    Parameters:
    min_n (int):         minimum number of samples used to compute a score
    max_n (int):         number of samples to generate per sample set
    qid (str):           the question ID
    dtype (str):         the data type of the sample solutions
    func_name (str):     the name of the function to be evaluated
    scorer (Scorer):     scorer used for measuring discrepancy
    M (int):             number of sample sets to generate
    max_parallel (int):  maximum allowable number of parallel processes
    save_samples (bool): save all Monte Carlo samples 
                         (not recommended; expensive for memory!)
    test_label (str):    name of the test case to be evaluated
    test_args (list):    arguments to be used for the test case
    proj_method (bool):  projection method in the case of multidimensional samples
                         
    Returns: 
    None
    """
    
    print('- - - - - MONTE CARLO SAMPLING - - - - -\n')
    
    # Generate IDs for Monte Carlo sampling and generate samples and scores
    print(f'Generating {M} sample sets of size {max_n} and computing their scores.')
    sids = [f'mc_solution_{i+1}' for i in range(M)]
    monte_carlo_sample_multi(
        sids, qid, min_n, max_n, dtype, func_name, max_parallel,
        test_label=test_label, test_args=test_args, save_samples=save_samples, 
        scorer=scorer, proj_method=proj_method
    )
    
    # Calculate the sizes for which scores were calculated
    sizes = []
    curr = min_n
    while curr < max_n:
        sizes.append(curr)
        curr *= 2
    sizes.append(max_n)
    
    # Accumulate the sorted scores into one file
    print('\nAccumulating scores across all sample sets.')
    scores = {size: [] for size in sizes}
    for sid in tqdm(sids):
        sid_scores = json.load(open(os.path.join(DATA_DIR, qid, 'solution', 'mc_solutions', sid, test_label,
                                                 f'mc_scores_{str(scorer)}{proj_method}.json'))
        )
        for size in sizes:
            scores[size].append(sid_scores[str(size)])
    print('Sorting scores.')
    for size in tqdm(scores):
        scores[size].sort()
    with open(os.path.join(
        DATA_DIR, qid, 'solution', 'mc_solutions', test_label, f'mc_scores_{str(scorer)}{proj_method}.json'), 'w'
             ) as f:
        json.dump(scores, f, indent=4)
    if not save_samples:
        print('Clearing individual score sets.')
        for sid in tqdm(sids):
            shutil.rmtree(os.path.join(DATA_DIR, qid, 'solution', 'mc_solutions', sid))
    print('Success!\n\n')
    

if __name__ == '__main__':
    
    # Load in command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('qid', type=str)
    parser.add_argument('file_path', type=str)
    parser.add_argument('func_name', type=str)
    
    # Model parameter arguments
    parser.add_argument('--dtype', type=str, default='scalar')
    parser.add_argument('--far', type=float, default=None)
    parser.add_argument('--frr', type=float, default=0.01)
    parser.add_argument('--max_n', type=int, default=409600)
    parser.add_argument('--min_n', type=int, default=400)
    parser.add_argument('--n_scale_factor', type=int, default=4)
    parser.add_argument('--n_soln_samples', type=int, default=500000)
    parser.add_argument('--scorer', type=str, default='AndersonDarlingScorer')
    parser.add_argument('--save_mc_samples', action='store_true')
    parser.add_argument('--M', type=int, default=1000)
    parser.add_argument('--max_parallel', type=int, default=20)
    parser.add_argument('--proj_method', type=str, default=None)
    
    args = parser.parse_args()
    qid = args.qid
    file_path = args.file_path

    # Load in the student and solution data
    load_qid_data(file_path, qid)
    
    # Save parameters to editable config.ini file
    config = configparser.ConfigParser()
    config['Parameters'] = {
        'dtype': args.dtype,
        'func_name': args.func_name,
        'frr': args.frr,
        'max_n': args.max_n,
        'min_n': args.min_n,
        'n_scale_factor': args.n_scale_factor,
        'n_soln_samples': args.n_soln_samples,        
        'scorer': args.scorer
    }
    if args.proj_method in ['ED', 'OP']:
        config['Parameters']['proj_method'] = args.proj_method
    elif 'array' in args.dtype and args.proj_method is not None:
        config['Parameters']['proj_method'] = 'ED'
                
    if args.far is not None: 
        config['Parameters']['far'] = args.far
    else:
        config['Parameters']['far'] = '0.01'
        
    config_path = os.path.join(DATA_DIR, qid, 'setup', 'config.ini')
    with open(config_path, 'w') as f:
        config.write(f)
    
    # Set model parameters to variables
    config = configparser.ConfigParser()
    if os.path.isfile(os.path.join(DATA_DIR, qid, 'setup', 'config.ini')):   
        config.read(os.path.join(DATA_DIR, qid, 'setup', 'config.ini'))
        
    dtype = config['Parameters']['dtype']
    func_name = config['Parameters']['func_name']
    max_n = int(config['Parameters']['max_n'])
    min_n = int(config['Parameters']['min_n'])
    n_soln_samples = int(config['Parameters']['n_soln_samples'])
    scorer_name = config['Parameters']['scorer']
    scorer_map = make_scorer_map()
    scorer = scorer_map[scorer_name]
    
    # Determine whether we use function arguments
    args_path = os.path.join(DATA_DIR, qid, 'setup', 'grading_arguments.json')
    if os.path.isfile(args_path):
        with open(os.path.join(args_path)) as f:
            test_label = 'grading_case'
            test_args = json.load(f)
    else:
        test_label = ''
        test_args = []

    # Sample from the solution program
    sample_solution(qid, n_soln_samples, dtype, func_name, test_label=test_label, test_args=test_args)
    
    # Generate Monte Carlo samples and scores
    if scorer.monte_carlo:
        monte_carlo(min_n, max_n, qid, dtype, func_name, scorer, M=args.M, max_parallel=args.max_parallel,
                   test_label=test_label, test_args=test_args, save_samples=args.save_mc_samples)