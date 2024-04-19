import argparse
import configparser
import json
import os
import shutil

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
    qid (str):                the question ID
    
    Returns: 
    None
    """
    
    print('\n- - - - - DATA PROCESSING - - - - -\n')
    # Create the question ID directory, if it doesn't exist
    print('Creating the question ID directory.')
    if not os.path.isdir(os.path.join(DATA_DIR, qid)):
        os.makedirs(os.path.join(DATA_DIR, qid))
        
    # Create the config file
    print('Creating config file in the question ID directory.')
    print('These parameters can be adjusted according to the problem being evaluated.\n')
    if not os.path.isfile(os.path.join(DATA_DIR, qid, 'config.ini')):
        create_config_file(os.path.join(DATA_DIR, qid, 'config.ini'))
    
    # Check the validity of the provided file path
    if not os.path.exists(file_path):
        raise ValueError(f'The file path "{file_path}" does not exist. Please enter a valid file path.')
    if not os.path.isdir(file_path):
        raise ValueError(f'Please ensure the file path "{file_path}" is a directory.')
    
    # Load solution program data
    print('Loading in the solution program data.')
    if 'solution.py' not in os.listdir(file_path):
        raise ValueError(f'Please create the solution program "solution.py" in the file path "{file_path}".')
    with open(os.path.join(file_path, 'solution.py')) as f:
        solution_program = f.read()
    
    # Load student program data
    print('Loading in the student program data.')
    student_programs = {}     
    for item in os.listdir(file_path): 
        if item == 'solution.py':
            continue
        if item.endswith('.json'):  # Student data formatted as a .json file
            if len(os.listdir(file_path)) > 2:
                raise ValueError(f'Student data must be formatted as a .json file or as separate [sid].py files.')
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
    
    # Create the student ID directories with their programs
    for sid in student_programs: 
        sid_dir = os.path.join(DATA_DIR, qid, 'students', sid)
        if not os.path.isdir(sid_dir):
            os.makedirs(sid_dir)
        with open(os.path.join(sid_dir, 'program.py'), 'w') as f:   
            f.write(student_programs[sid])
    
    # Create results directory
    print('Creating results directory.')
    if not os.path.isdir(os.path.join(DATA_DIR, qid, 'results')):
        os.makedirs(os.path.join(DATA_DIR, qid, 'results'))
    print('Success!\n')
    
    
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
    
    print('\n- - - - - SOLUTION SAMPLE GENERATION - - - - -\n')
    
    # Generate the solution samples
    print(f'Generating {n_samples} samples from the solution program.')
    sid = 'solution'
    sample_sid_single(sid, qid, n_samples, dtype, func_name, test_label=test_label, test_args=test_args)  
    sample_path = os.path.join(DATA_DIR, qid, 'solution', sid, test_label, 'samples.npy')
    samples = np.load(sample_path, allow_pickle=True)
    print(f'Succesfully generated {len(samples)} solution samples!\n')
    
    
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
    
    # TODO: Change M back to 1000
    
    print('\n- - - - - MONTE CARLO SAMPLING - - - - -\n')
    
    # Generate IDs for Monte Carlo sampling and generate samples and scores
    print(f'Generating {M} sample sets of size {max_n} and computing their scores.')
    sids = [f'mc_solution_{i+1}' for i in range(M)]
    monte_carlo_sample_multi(
        sids, qid, min_n, max_n, dtype, func_name, max_parallel,
        test_label='', test_args=[], save_samples=save_samples, 
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
        sid_scores = json.load(open(os.path.join(DATA_DIR, qid, 'solution', 'mc_solutions', sid, 
                                                 f'mc_scores_{str(scorer)}{proj_method}.json'))
        )
        for size in sizes:
            scores[size].append(sid_scores[str(size)])
    print('Sorting scores.')
    for size in tqdm(scores):
        scores[size].sort()
    with open(os.path.join(
        DATA_DIR, qid, 'solution', 'mc_solutions', f'mc_scores_{str(scorer)}{proj_method}.json'), 'w'
             ) as f:
        json.dump(scores, f)
    if not save_samples:
        print('Clearing individual score sets.')
        for sid in tqdm(sids):
            shutil.rmtree(os.path.join(DATA_DIR, qid, 'solution', 'mc_solutions', sid))
    print('Success!\n')
    

def create_config_file(file_path):
    """
    Creates a config.ini file with the specified contents at the given file path.
    
    file_path (str): the path at which the config file will be created
    """
    
    config = configparser.ConfigParser()
    
    print('CONFIG SETUP\n')
    
    print('Select the data type.')
    print('(1) scalar:   for single values')
    print('(2) list:     for lists (concatenated into one distribution)')
    print('(3) multidim: for higher dimensional arrays\n')
    dtype = input('Select one of the options: ')
    while dtype not in ['1', '2', '3']:
        dtype = input('Please select a valid option (1, 2, or 3): ')
        
    if dtype == '1':
        dtype = 'scalar'
    elif dtype == '2':
        dtype = 'list'
    else:
        print('\nEnter the shape of the output array for the problem.')
        print('e.g. (3, 5) for output of shape (3, 5)')
        shape = input('Shape tuple: ')
        # TODO: Check if it's a valid shape tuple, etc
        dtype = f'array_shape_{shape}'
        
    print('\nSelect the false rejection rate.')
    print('This is a float, e.g. 0.01.')
    frr = input('FRR: ')
    
    print('\nSelect the number of solution samples.')
    print('This is an int, e.g. 500000.')
    num_soln_samples = input('Number of samples: ')
    
    print('\nSelect the minimum number of student samples used for grading.')
    print('This is an int, e.g. 200.')
    min_n = input('Minimum number of samples: ')
    
    print('\nSelect the maximum number of student samples used for grading.')
    print('This is an int, e.g. 500000.')
    print('Note: the choose_n.py script can help to determine this value.')
    max_n = input('Maximum number of samples: ')
    
    print('\nSelect the scoring function.')
    print('Pre-existing options: TScorer, MSDScorer, AndersonDarlingScorer, WassersteinScorer.')
    scorer = input('Scorer: ')
    
    print('\nSelect the maximum number of parallel processes for sampling.')
    print('This is an int, e.g. 50')
    max_parallel = input('Max parallel: ')
    
    print('\nWrite the name of the function to be executed for grading.')
    print('This is a string, e.g. myFunction')
    func_name = input('Function name: ')
    
    # Set the parameters in the file
    config['Parameters'] = {
        'dtype': dtype,
        'frr': frr, 
        'num_soln_samples': num_soln_samples,
        'min_n': min_n,
        'max_n': max_n,
        'scorer': scorer, 
        'max_parallel': max_parallel,
        'func_name': func_name
    }
    
    print('\n')
    with open(file_path, 'w') as f:
        config.write(f)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('qid', type=str)
    parser.add_argument('file_path', type=str)
    args = parser.parse_args()
    
    qid = args.qid
    file_path = args.file_path
    
    # Load in the student and solution data
    load_qid_data(file_path, qid)
    
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

    # Sample from the solution program
    sample_solution(qid, num_soln_samples, dtype, func_name)
    
    # Generate Monte Carlo samples and scores
    monte_carlo(min_n, max_n, qid, dtype, func_name, scorer, M=10)