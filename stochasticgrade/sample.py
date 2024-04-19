import argparse
import ast
import contextlib
import io
import itertools
import json
import multiprocessing as mp
import numpy as np
import os
import pickle
import shutil
import signal
import sys
import time
import warnings

from tqdm import tqdm
from multiprocessing import Process, Queue
from numbers import Number

from stochasticgrade.constants import DATA_DIR
from stochasticgrade.utils import get_euclidean_distance, get_orthogonal_projection


# Suppress syntax warnings
warnings.filterwarnings('ignore')


def get_samples(
    sid, qid, num_samples, dtype, func_name, test_label='', test_args=[],
    early_stopping=10000, max_timeouts=5, append_samples=False, pos=0
):
    """
    Sample `num_samples` samples from `prog`. 
    dtype is either scalar or list.
    """
    
    # Load the program from which we will be sampling. If samples already exist,
    # load the samples in as well. 
    if 'solution' in sid:
        sample_path = os.path.join(DATA_DIR, qid, 'solution', sid, test_label, 'samples.npy')
        prog_path = os.path.join(DATA_DIR, qid, 'solution', 'solution', 'program.py')
    elif 'closest_error' in sid:
        sample_path = os.path.join(DATA_DIR, qid, 'closest_error', sid, test_label, 'samples.npy')
        prog_path = os.path.join(DATA_DIR, qid, 'closest_error', sid, 'program.py')
    else:
        sample_path = os.path.join(DATA_DIR, qid, 'students', sid, test_label, 'samples.npy')
        prog_path = os.path.join(DATA_DIR, qid, 'students', sid, 'program.py')
    with open(prog_path) as f:
        prog = f.read()
        
    if append_samples and os.path.isfile(sample_path):
        samples = list(np.load(sample_path, allow_pickle=True))
    else:
        samples = []     
    
    if dtype == 'scalar':
        sample_fn = scalar_sample 
    elif dtype == 'list':
        sample_fn = list_sample
    elif 'array_shape_' in dtype:
        sample_fn = multidim_sample
        
    # Set up progress bar and error handling
    pid = os.getpid()
    pbar = tqdm(range(num_samples), leave=False, position=pos,
                dynamic_ncols=True, nrows=20, postfix=f'{pid}')
    def sigterm_handler(_signo, _stack_frame):
        pbar.close()
        return None, []
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    timeout_cnt = 0
    samples_remaining = num_samples
    
    # Begin sampling
    while samples_remaining > 0:
        
        # Check for early stopping (timeouts or degenerate distributions)
        if timeout_cnt > max_timeouts:
            samples = []
            break
        if len(samples) == early_stopping and len(np.unique(samples)) == 1:
            samples = [samples[0]]
            break
        
        # Proceed with sampling as normal
        
        start = time.time()
        val = sample_fn(qid, prog, dtype, func_name, test_args=test_args)
        end = time.time()
        
        # Add the newly sampled value(s) to the list of samples
        if dtype == 'scalar':
            if val is None:
                timeout_cnt += 1
                break
            samples.append(val)
                
        if dtype == 'list':
            if val is None or len(val) == 0:
                timeout_cnt += 1
                break
            if not isinstance(val[0], Number):
                break
            val = list(val)
            samples.extend(val)
            
        if 'array_shape_' in dtype: 
            shape = tuple([int(i) for i in dtype.split('array_shape_')[1][1:-1].split(',')])
            if val is None or len(val) == 0:
                timeout_cnt += 1
                break
            if np.array(val).shape != shape:
                break
            samples.append(val)
         
        # Update the progress bar if the sample is valid
        if val is not None:
            if dtype == 'scalar':
                pbar.update(1)
                samples_remaining -= 1
            if dtype == 'list':
                pbar.update(min(len(val), samples_remaining))
                samples_remaining -= len(val)
            if 'array_shape_' in dtype:
                pbar.update(1)
                samples_remaining -= 1
    
    pbar.close()
    return samples


def scalar_sample(qid, prog, dtype, func_name, test_args=[]):
    """
    Execution function for scalar data types.
    """
    return exec_program(qid, prog, func_name, test_args=test_args)


def list_sample(qid, prog, dtype, func_name, test_args=[]):
    """
    Execution function for list data type.
    """
    return exec_program(qid, prog, func_name, test_args=test_args, allowed_types=[list])


def multidim_sample(qid, prog, dtype, func_name, test_args=[]):
    """
    Execution function for multidimensional data types.
    """
    return exec_program(qid, prog, func_name, test_args=test_args, allowed_types=[list])


def evaluate_student_code(qid, prog, func_name, test_args=[]):
    """
    Evaluate the student code in the context of the associated problem
    environment. Suppress stdout.
    """
    
    val = None
    
    try:
        # Redirect output to prevent extraneous printing
        with contextlib.redirect_stdout(io.StringIO()):
            # Execute the student code and record the output
            local_scope = {}
            exec(prog, local_scope, local_scope)
            func = local_scope.get(func_name)
            print(func)
            if callable(func):
                val = func(*test_args)
            else:
                raise ValueError(f'No function named "{func_name}" found in the code for question {qid}')
    except Exception as e:
        print(e)
        pass
    
    return val


def _alarm_handler(signum, frame):
    raise TimeoutError
    

def exec_program(qid, prog, func_name, timeout=1, test_args=[], allowed_types=[]):
    """
    Evaluate the student program and return its return value.
    Return None if student program cannot be evaluated.
    """

    val = None
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)

    try:  # Attempt to run the program
        val = evaluate_student_code(qid, prog, func_name, test_args=test_args)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception:  # Caught an error
        return None
    finally:  # Reset alarm
        signal.alarm(0)

    if not isinstance(val, tuple([Number] + allowed_types)):
        return None
    
    return val


def sample_sid_single(
    sid, qid, num_samples, dtype, func_name,
    test_label='', test_args=[], append_samples=False, pos=0, proc_queue=None
):
    """
    Sampling process for singlethreaded sampling. 
    """
    
    # Create the appropriate directory name where samples will be stored
    dir_lst = [DATA_DIR, qid]
    if 'solution' in sid:
        sid_type = 'solution'
    elif 'closest_error' in sid:
        sid_type = 'closest_error'
    else:
        sid_type = 'students'
    dir_lst.append(sid_type)
    dir_lst += [sid]
    dir_lst += [test_label]
    dirname = os.path.join(*dir_lst)
    sample_path = os.path.join(dirname, 'samples.npy')
    
    # If samples already exist or we don't want to append, don't do anything
    if not os.path.isfile(sample_path) or append_samples:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        samples = get_samples(
            sid, qid, num_samples, dtype, func_name, test_label=test_label, 
            test_args=test_args, append_samples=append_samples, pos=pos
        )
        
        np.save(sample_path, np.array(samples))    
        
    if proc_queue is not None:
        proc_queue.put((mp.current_process().pid, pos))
        
        
def sample_sid_multi(
    sids, qid, num_samples, dtype, func_name, max_parallel,
    test_label='', test_args=[], append_samples=False
):

    # Set up parallelization
    proc_dict = dict()
    proc_queue = Queue(max_parallel)
    processes = []
    def _deque_proc():
        (old_pid, old_pos) = proc_queue.get()
        old_proc = proc_dict[old_pid]
        old_proc.join()
        old_proc.close()
        return old_pos
    filled_pos = 0
    pbar = tqdm(sids, total=len(sids), dynamic_ncols=True, nrows=20)

    # Begin sampling for each sid
    for sid in pbar:
        if filled_pos >= max_parallel:
            getpos = _deque_proc()
        else:
            pos = filled_pos % max_parallel 
        filled_pos += 1
        p = Process(target=sample_sid_single,
                    args=[sid, qid, num_samples, dtype, func_name, test_label, test_args,
                          append_samples, save_samples, pos, proc_queue])
        p.start()
        proc_dict[p.pid] = p
        processes.append(p)
        
    # Wait for all processes to complete
    for p in processes:
        p.join()

    while not proc_queue.empty():
        _deque_proc()
        
        
def monte_carlo_sample_single(
    sid, qid, min_n, max_n, dtype, func_name,
    test_label='', test_args=[], append_samples=False, pos=0, proc_queue=None, 
    save_samples=False, scorer=None, proj_method=None
):
    """
    Sampling process for singlethreaded Monte Carlo sampling. 
    """
    
    # Create the appropriate directory name where samples will be stored
    dir_lst = [DATA_DIR, qid]
    dir_lst.append('solution')
    dir_lst.append('mc_solutions')
    dir_lst += [sid]
    dir_lst += [test_label]
    dirname = os.path.join(*dir_lst)
    sample_path = os.path.join(dirname, 'samples.npy')
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    
    # If samples already exist or we don't want to append, don't do anything
    if not os.path.isfile(sample_path) or append_samples:

        samples = get_samples(
            sid, qid, max_n, dtype, func_name, test_label=test_label, 
            test_args=test_args, append_samples=append_samples, pos=pos
        )
        
        if save_samples:
            np.save(sample_path, np.array(samples)) 
        
    # Calculate the score for the samples
    # The scores are calculated for each sample size in powers of 2 from `min_n` to `max_n`
    sizes = []
    curr = min_n
    while curr < max_n:
        sizes.append(curr)
        curr *= 2
    sizes.append(max_n)
    
    save_path = os.path.join(dirname, f'mc_scores_{str(scorer)}{proj_method}.json')
    scores = {}
    soln_samples = np.load(os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 'samples.npy'))
    
    # Determine whether to use the samples themselves (1D samples) or their projections
    # (multidimensional samples) for computing the score
    if 'array_shape' in dtype:
        if proj_method == 'ED':
            stud_dists = get_euclidean_distance(samples, soln_samples, sid, qid)
            soln_dists = get_euclidean_distance(soln_samples, soln_samples, sid, qid)
        else:
            stud_dists = get_orthogonal_projection(samples, soln_samples, sid, qid)
            soln_dists = get_orthogonal_projection(soln_samples, soln_samples, sid, qid)
    else:
        stud_dists = samples
        soln_dists = soln_samples
            
    for size in sizes:
        np.random.shuffle(stud_dists)
        np.random.shuffle(soln_dists)
        score = scorer.compute_score(stud_dists[:size], soln_dists)
        scores[size] = score
    with open(save_path, 'w') as f:
        json.dump(scores, f)
        
    if proc_queue is not None:
        proc_queue.put((mp.current_process().pid, pos))
        
        
def monte_carlo_sample_multi(
        sids, qid, min_n, max_n, dtype, func_name, max_parallel,
        test_label='', test_args=[], append_samples=False, save_samples=False, 
        scorer=None, proj_method=''
):
    
    # Set up parallelization
    proc_dict = dict()
    processes = []
    proc_queue = Queue(max_parallel)
    def _deque_proc():
        (old_pid, old_pos) = proc_queue.get()
        old_proc = proc_dict[old_pid]
        old_proc.join()
        old_proc.close()
        return old_pos
    filled_pos = 0
    pbar = tqdm(sids, total=len(sids), dynamic_ncols=True, nrows=20)

    # Begin sampling for each sid
    for sid in pbar:
        if filled_pos >= max_parallel:
            getpos = _deque_proc()
        else:
            pos = filled_pos % max_parallel
        filled_pos += 1 
        p = Process(target=monte_carlo_sample_single,
                    args=[sid, qid, min_n, max_n, dtype, func_name, test_label, test_args,
                          append_samples, pos, proc_queue, save_samples, scorer, proj_method])
        p.start()
        proc_dict[p.pid] = p
        processes.append(p) 

    # Wait for all processes to complete
    for p in processes:
        p.join()

    while not proc_queue.empty():
        _deque_proc()
            
    
def get_test_suite(qid, num_tests=20, all_test_suites=False):
    """
    Create test suite cases for problems in which multiple sample sets
    from the same program are useful.
    """
    
    test_suite_dir = os.path.join(DATA_DIR, qid, 'test_suites')
    if not os.path.isdir(test_suite_dir):
        os.mkdir(test_suite_dir)
        print(f'Must initialize the {qid}.py file containing the', 
              'generate_test_suites() function in the test suite directory.')
        print('If you would like to specify a subset of cases, do so', 
              f'in a {qid}.labels.json file in the test suite directory.')
        raise Exception
        
    
    suite_fname = os.path.join(test_suite_dir, f'{qid}.pkl')

    if not os.path.isfile(suite_fname):
        gen_fname = os.path.join(test_suite_dir, f'{qid}.py')
        with open(gen_fname) as f:
            script = f.read()
        exec(script, locals(), locals())

        test_suites = {}
        for i in range(num_tests):
            test_case = locals()['generate_test_suites']()
            test_label = f'case_{i}'
            test_suites[test_label] = test_case

        with open(suite_fname, 'wb') as f:
            pickle.dump(test_suites, f)
    else:
        print('loading existing test cases...')
        with open(suite_fname, 'rb') as f:
            test_suites = pickle.load(f)

    if not all_test_suites:
        labels_fname = os.path.join(test_suite_dir, f'{qid}.labels.json')
        chosen_labels = json.load(open(labels_fname))
        print('testing a subset of test cases: ', chosen_labels)
        test_suites = dict((k,v) for k,v in test_suites.items()
                           if k in chosen_labels)
    return test_suites
                                    