import contextlib
import io
import json
import multiprocessing as mp
import numpy as np
import os
import signal
import time
import warnings

from tqdm import tqdm
from multiprocessing import Process, Queue
from numbers import Number

from stochasticgrade.constants import DATA_DIR
from stochasticgrade.utils import get_euclidean_distances, get_orthogonal_projections


# Suppress syntax warnings
warnings.filterwarnings('ignore')


def get_samples(
    sid, qid, num_samples, dtype, func_name, test_label='', test_args=[],
    early_stopping=10000, max_timeouts=5, append_samples=False, pos=0, sample_to=None,
    show_progress=True, long_sampling_time=300
):
    """
    Samples `num_samples` samples from the student program belonging to `sid`. 

    Parameters:
    sid (str):             the student ID
    qid (str):             the question ID
    num_samples (int):     the number of samples to be generated
    dtype (str):           the data type of the program output
    func_name (str):       the function to be sampled
    test_label (str):      the label of the function arguments
    test_args (list):      the function arguments needed for execution
    early_stopping (int):  early termination criteria
    max_timeouts (int):    number of permissible timeouts before termination
    append_samples (bool): whether to append generated samples to existing samples
    pos (int):             progress bar position
    sample_to (int):       sample exactly to the provided number

    Returns: 
    samples (list): the generated samples
    """
    
    # Load the program from which we will be sampling. If samples already exist,
    # load the samples in as well. 
    if 'solution' in sid:
        sample_path = os.path.join(DATA_DIR, qid, 'solution', sid, test_label, 'samples.npy')
        prog_path = os.path.join(DATA_DIR, qid, 'solution', 'solution', 'program.py')
    elif 'closest_error' in sid:
        sample_path = os.path.join(DATA_DIR, qid, 'setup', sid, test_label, 'samples.npy')
        prog_path = os.path.join(DATA_DIR, qid, 'setup', sid, 'program.py')
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
    elif 'array' in dtype:
        sample_fn = multidim_sample
        
    # Set up progress bar and error handling
    pid = os.getpid()

    if show_progress: 
        pbar = tqdm(range(num_samples), leave=False, position=pos,
                dynamic_ncols=True, nrows=20, postfix=f'{pid}')
    else:
        pbar = None
    def sigterm_handler(_signo, _stack_frame):
        if pbar:
            pbar.close()
        return None, []
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    timeout_cnt = 0
    samples_remaining = num_samples if sample_to is None else sample_to - len(samples)

    sampling_time = time.time()
    
    # Begin sampling
    while samples_remaining > 0:

        if time.time() - sampling_time > long_sampling_time:
            print(f'Still sampling for sid {sid} after {time.time() - sampling_time:.1f} seconds. Consider inspecting the program.')
            sampling_time = time.time()
        
        # Check for early stopping (timeouts or degenerate distributions)
        if timeout_cnt > max_timeouts:
            samples = []
            break
        if len(samples) == early_stopping and len(np.unique(samples)) == 1:
            samples = [samples[0]]
            break
        
        # Proceed with sampling as normal
        
        start = time.time()
        val = sample_fn(sid, qid, prog, func_name, test_args=test_args)
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
            
        if 'array' in dtype: 
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
                if pbar:
                    pbar.update(1)
                samples_remaining -= 1
            if dtype == 'list':
                if pbar:
                    pbar.update(min(len(val), samples_remaining))
                samples_remaining -= len(val)
            if 'array' in dtype:
                if pbar:
                    pbar.update(1)
                samples_remaining -= 1
    
    if pbar:
        pbar.close()
    return samples


def scalar_sample(sid, qid, prog, func_name, test_args=[]):
    """
    Execution function for scalar data types.

    Parameters:
    sid (str):        the student ID
    qid (str):        the question ID
    prog (str):       the sampled program
    func_name (str):  the function to be sampled
    test_args (list): the function arguments needed for execution

    Returns: 
    A single sample, obtained from `exec_program`
    """
    return exec_program(sid, qid, prog, func_name, test_args=test_args)


def list_sample(sid, qid, prog, func_name, test_args=[]):
    """
    Execution function for list data type.

    Parameters:
    sid (str):        the student ID
    qid (str):        the question ID
    prog (str):       the sampled program
    func_name (str):  the function to be sampled
    test_args (list): the function arguments needed for execution

    Returns: 
    A single sample, obtained from `exec_program`
    """
    return exec_program(sid, qid, prog, func_name, test_args=test_args, allowed_types=[list])


def multidim_sample(sid, qid, prog, func_name, test_args=[]):
    """
    Execution function for multidimensional data types.

    Parameters:
    sid (str):        the student ID
    qid (str):        the question ID
    prog (str):       the sampled program
    func_name (str):  the function to be sampled
    test_args (list): the function arguments needed for execution

    Returns: 
    A single sample, obtained from `exec_program`
    """
    return exec_program(sid, qid, prog, func_name, test_args=test_args, allowed_types=[list])


def evaluate_student_code(sid, qid, prog, func_name, test_args=[]):
    """
    Evaluate the student code in the context of the associated problem
    environment. This generates a single sample if successful.

    Parameters:
    qid (str):        the question ID
    prog (str):       the sampled program
    func_name (str):  the function to be sampled
    test_args (list): the function arguments needed for execution

    Returns: 
    val: a single sample
    """

    val = None
    try:
        # Redirect output to prevent extraneous printing
        with contextlib.redirect_stdout(io.StringIO()):
            # Execute the student code and record the output
            local_scope = {}
            exec(prog, local_scope, local_scope)
            func = local_scope.get(func_name)
            if callable(func):
                val = func(*test_args)
            else:
                raise ValueError(f'No function named "{func_name}" found in the code for question {qid}')
    except Exception as e:
        # Save the error
        path = os.path.join(DATA_DIR, qid, 'results', 'inexecutable_sids.json')
        if os.path.isfile(path):
            with open(path) as f:
                progs = json.load(path)
        else:
            progs = {}
        progs[sid] = e
        with open(path, 'w') as f:
            json.dump(progs, f, indent=4)
    
    return val


def _alarm_handler(signum, frame):
    """
    Timeout handler.
    """
    raise TimeoutError
    

def exec_program(sid, qid, prog, func_name, timeout=1, test_args=[], allowed_types=[]):
    """
    Evaluate the student program and return its return value.
    Return None if student program cannot be evaluated.

    Parameters:
    sid (str):            the student ID
    qid (str):            the question ID
    prog (str):           the sampled program
    func_name (str):      the function to be sampled
    timeout (int):        the maximum amount of time to wait for a single sample
    test_args (list):     the function arguments needed for execution
    allowed_types (list): legal return value types

    Returns:
    val: a single sample
    """

    val = None
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)

    try:  # Attempt to run the program
        val = evaluate_student_code(sid, qid, prog, func_name, test_args=test_args)
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
    sid, qid, num_samples, dtype, func_name, test_label='', test_args=[], 
    append_samples=False, pos=0, proc_queue=None, proj_method='ED', sample_to=None,
    show_progress=True
):
    """
    Sampling process for singlethreaded sampling. Generates `num_samples` samples
    for the program belonging to `sid`.

    Parameters:
    sid (str):             the student ID
    qid (str):             the question ID
    num_samples (int):     the number of samples to be generated
    dtype (str):           the data type of the program output
    func_name (str):       the function to be sampled
    test_label (str):      the label of the function arguments
    test_args (list):      the function arguments needed for execution
    append_samples (bool): whether to append generated samples to existing samples
    pos (int):             progress bar position
    proc_queue (Queue):    the process queue
    proj_method (str):     the projection method used for multidimensional samples
    sample_to (int):       sample exactly to the provided number

    Returns: 
    None
    """
    
    # Create the appropriate directory name where samples will be stored
    if 'solution' in sid:
        sid_type = 'solution'
    elif 'closest_error' in sid:
        sid_type = 'setup'
    else:
        sid_type = 'students'
    dirname = os.path.join(DATA_DIR, qid, sid_type, sid, test_label)
    sample_path = os.path.join(dirname, 'samples.npy')
    
    # If samples already exist or we don't want to append, don't do anything
    if not os.path.isfile(sample_path) or append_samples:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        samples = get_samples(
            sid, qid, num_samples, dtype, func_name, test_label=test_label, 
            test_args=test_args, append_samples=append_samples, pos=pos, sample_to=sample_to,
            show_progress=show_progress
        )
        
        # Save the samples
        np.save(sample_path, np.array(samples))    
        
        # If our data is multidimensional, calculate and save the projections
        # This projection is specified by proj_method
        if 'array' in dtype and samples:
            samples = np.array(samples)
            soln_samples = np.load(os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 'samples.npy'))
            if proj_method == 'OP':  # Orthogonal Projection method
                stud_dists = get_orthogonal_projections(samples, soln_samples, sid, qid, test_label)
            else:  # Assuming Euclidean Distance method
                stud_dists = get_euclidean_distances(samples, soln_samples, sid, qid, test_label)
        
    if proc_queue is not None:
        proc_queue.put((mp.current_process().pid, pos))
        
        
def sample_sid_multi(
    sids, qid, num_samples, dtype, func_name, max_parallel,
    test_label='', test_args=[], append_samples=False, proj_method='ED', sample_to=None,
    show_progress=True
):
    """
    Sampling process for multithreaded sampling. Generates `num_samples` samples
    for the programs belonging to `sids`.

    Parameters:
    sids (list):           a list of student IDs
    qid (str):             the question ID
    num_samples (int):     the number of samples to be generated
    dtype (str):           the data type of the program output
    func_name (str):       the function to be sampled
    max_parallel (int):    the maximum number of parallel sampling processes
    test_label (str):      the label of the function arguments
    test_args (list):      the function arguments needed for execution
    append_samples (bool): whether to append generated samples to existing samples
    proj_method (str):     the projection method used for multidimensional samples
    sample_to (int):       sample exactly to the provided number

    Returns: 
    None
    """

    # Set up parallelization
    proc_dict = dict()
    proc_queue = Queue(max_parallel)
    filled_pos = 0
    pbar = tqdm(sids, total=len(sids), dynamic_ncols=True, nrows=20)

    # Handle dequeueing 
    def _dequeue_proc():
        (old_pid, old_pos) = proc_queue.get()
        old_proc = proc_dict[old_pid]
        old_proc.join()
        old_proc.close()
        pbar.update(1)  
        return old_pos

    # Begin sampling for each sid
    for sid in sids:
        if filled_pos >= max_parallel:
            pos = _dequeue_proc()
        else:
            pos = filled_pos + 1
        filled_pos += 1
        p = Process(target=sample_sid_single,
                    args=[sid, qid, num_samples, dtype, func_name, test_label, test_args,
                          append_samples, pos, proc_queue, proj_method, sample_to, show_progress])
        p.start()
        proc_dict[p.pid] = p

    for _ in range(max_parallel):
        _dequeue_proc()
        
        
def monte_carlo_sample_single(
    sid, qid, min_n, max_n, dtype, func_name,
    test_label='', test_args=[], append_samples=False, pos=0, proc_queue=None, 
    save_samples=False, scorer=None, proj_method='ED'
):
    """
    Sampling process for singlethreaded Monte Carlo sampling. Generates 
    samples for the program belonging to `sid`. Evaluates scores for various 
    sample sizes calculated between `min_n` and `max_n`.

    Parameters:
    sid (str):             the student ID
    qid (str):             the question ID
    min_n (int):           the minimum number of samples to score
    max_n (int):           the maximum number of samples to score
    dtype (str):           the data type of the program output
    func_name (str):       the function to be sampled
    test_label (str):      the label of the function arguments
    test_args (list):      the function arguments needed for execution
    append_samples (bool): whether to append generated samples to existing samples
    pos (int):             progress bar position
    proc_queue (Queue):    the process queue
    save_samples (bool):   keep Monte Carlo samples
    scorer (Scorer):       scoring function
    proj_method (str):     projection method for multidimensional samples

    Returns: 
    None
    """
    
    # Create the appropriate directory name where samples will be stored
    dirname = os.path.join(DATA_DIR, qid, 'solution', 'mc_solutions', sid, test_label)
    sample_path = os.path.join(dirname, 'samples.npy')
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    
    # If samples already exist and we don't want to append, don't sample
    if os.path.isfile(sample_path) and not append_samples:
        samples = np.load(sample_path)
    else:
        samples = get_samples(
            sid, qid, max_n, dtype, func_name, test_label=test_label, 
            test_args=test_args, append_samples=append_samples, pos=pos
        )
        samples = np.array(samples)
        
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
    if 'array' in dtype:
        if proj_method == 'OP':  # Assuming Orthogonal Projection method
            stud_dists = get_orthogonal_projections(samples, soln_samples, sid, qid, test_label)
            soln_dists = get_orthogonal_projections(soln_samples, soln_samples, sid, qid, test_label)
        else:  # Assuming Euclidean Distance method
            stud_dists = get_euclidean_distances(samples, soln_samples, sid, qid, test_label)
            soln_dists = get_euclidean_distances(soln_samples, soln_samples, sid, qid, test_label)
    else:
        stud_dists = samples
        soln_dists = soln_samples
            
    for size in sizes:
        np.random.shuffle(stud_dists)
        np.random.shuffle(soln_dists)
        score = scorer.compute_score(stud_dists[:size], soln_dists)
        scores[size] = score
    with open(save_path, 'w') as f:
        json.dump(scores, f, indent=4)
        
    if proc_queue is not None:
        proc_queue.put((mp.current_process().pid, pos))
        
        
def monte_carlo_sample_multi(
        sids, qid, min_n, max_n, dtype, func_name, max_parallel,
        test_label='', test_args=[], append_samples=False, save_samples=False, 
        scorer=None, proj_method='ED'
):
    """
    Sampling process for multithreaded Monte Carlo sampling. Generates 
    samples for the programs belonging to `sids`. Evaluates scores for various 
    sample sizes calculated between `min_n` and `max_n`.

    Parameters:
    sids (list):           list of student IDs
    qid (str):             the question ID
    min_n (int):           the minimum number of samples to score
    max_n (int):           the maximum number of samples to score
    dtype (str):           the data type of the program output
    func_name (str):       the function to be sampled
    max_parallel (int):    maximum number of parallel sampling processes
    test_label (str):      the label of the function arguments
    test_args (list):      the function arguments needed for execution
    append_samples (bool): whether to append generated samples to existing samples
    save_samples (bool):   keep Monte Carlo samples
    scorer (Scorer):       scoring function
    proj_method (str):     projection method for multidimensional samples

    Returns: 
    None
    """

    # Set up parallelization
    proc_dict = dict()
    proc_queue = Queue(max_parallel)
    filled_pos = 0
    pbar = tqdm(sids, total=len(sids), dynamic_ncols=True, nrows=20)

    # Handle dequeueing 
    def _dequeue_proc():
        (old_pid, old_pos) = proc_queue.get()
        old_proc = proc_dict[old_pid]
        old_proc.join()
        old_proc.close()
        pbar.update(1)  
        return old_pos

    # Begin sampling for each sid
    for sid in sids:
        if filled_pos >= max_parallel:
            pos = _dequeue_proc()
        else:
            pos = filled_pos + 1
        filled_pos += 1
        p = Process(target=monte_carlo_sample_single, args=(
                    sid, qid, min_n, max_n, dtype, func_name, test_label, test_args,
                    append_samples, pos, proc_queue, save_samples, scorer, proj_method))
        p.start()
        proc_dict[p.pid] = p

    for _ in range(max_parallel):
        _dequeue_proc()
            