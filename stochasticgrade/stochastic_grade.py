"""
stochastic_grade.py
-------------------
This file contains the grading algorithm, StochasticGrade.
It allows the evaluation of student programs. 
"""

import argparse
import configparser
import json
import multiprocessing as mp
import numpy as np
import os
import time

from tqdm import tqdm

from stochasticgrade.constants import DATA_DIR
from stochasticgrade.sample import sample_sid_single
from stochasticgrade.score import *


class StochasticGrade():
    """
    Implements the StochasticGrade algorithm.
    This allows one to grade student programs. 
    """
    
    def __init__(self, qid, scorer, sample_sizes, frr, dtype, func_name, n_soln_samples, proj_method='ED'):
        """
        Instantiate the algorithm. Gather the solution samples.
        
        Parameters:
        qid (str):                  the question ID
        scorer (Scorer):            the scoring function to be used
        sample_sizes (list of int): the sample sizes used for grading
        frr (float):                the false rejection rate
        dtype (str):                the data type of the program output
        func_name (str):            the function to be sampled
        n_soln_samples (int):       the number of solution samples
        proj_method (str):          the projection method used for multidimensional samples

        Returns: 
        None
        """
        self.qid = qid
        self.scorer = scorer
        self.sample_sizes = sample_sizes
        self.frr = frr
        self.dtype = dtype
        self.func_name = func_name
        self.n_soln_samples = n_soln_samples
        self.proj_method = proj_method
        
        # Check if the function requires input arguments
        args_path = os.path.join(DATA_DIR, qid, 'setup', 'grading_arguments.json')
        if os.path.isfile(args_path):
            self.test_label = 'grading_case'
            with open(args_path) as f:
                test_args = json.load(f)
                self.test_args = test_args[self.test_label]
        else:
            self.test_label = ''
            self.test_args = []
        
        # Load in solution samples or projections
        soln_sample_path = os.path.join(DATA_DIR, qid, 'solution', 'solution', self.test_label, 'samples.npy')
        if os.path.isfile(soln_sample_path):
            if 'array' not in self.dtype: 
                self.soln_samples = np.load(soln_sample_path, allow_pickle=True)
            else:
                proj_file = 'orthogonal_projections.npy' if self.proj_method == 'OP' else 'euclidean_distances.npy'
                soln_sample_path = os.path.join(DATA_DIR, self.qid, 'solution', 'solution', 
                                                self.test_label, proj_file)
                self.soln_samples = np.load(soln_sample_path, allow_pickle=True)
        else:
            raise ValueError(f'No solution samples found for {qid}! Did you run preprocess.py?')
        np.random.shuffle(self.soln_samples)
        
    def sample(self, n, sid, pos=0, show_sample_progress=True):
        """
        Expand the collection of total student samples to n.
        
        Parameters:
        sid (str):                   the student ID
        pos (int):                   the progress bar position
        show_sample_progress (bool): show the sample progress bar
        
        Returns:
        None
        """      
        
        # Determine the directory in which samples are saved
        if 'solution' in sid:
            sid_type = 'solution'
        elif 'closest_error' in sid:
            sid_type = 'setup'
        else:
            sid_type = 'students'
            
        file_path = os.path.join(DATA_DIR, self.qid, sid_type, sid, self.test_label, 'samples.npy')
        if not os.path.isfile(file_path):
            n_samples = n
        else:
            self.stud_samples = np.load(file_path, allow_pickle=True)
            n_samples = n - len(self.stud_samples)
        
        if n_samples > 0:      
            sample_sid_single(
                sid, self.qid, n_samples, self.dtype, self.func_name, append_samples=True, 
                pos=pos, test_label=self.test_label, test_args=self.test_args, show_progress=show_sample_progress
            )  
            
        # Load in samples
        self.stud_samples = np.load(file_path, allow_pickle=True)
        np.random.shuffle(self.stud_samples)
         
    def grade(self, sid, pos=0, delete_samples=False, show_sample_progress=True):
        """
        Grade the student program corresponding to the given student ID.
        Single process as part of the grade_parallel function.
        
        Parameters:
        sid (str):                   the student ID
        pos (int):                   progress bar position
        delete_samples (bool):       whether to keep or discard collected samples
        show_sample_progress (bool): show the sample progress bar
        
        Returns: 
        result (tuple): a tuple with grading results
        """ 
        
        # Determine the directory in which samples are saved
        if 'solution' in sid:
            sid_type = 'solution'
        elif 'closest_error' in sid:
            sid_type = 'setup'
        else:
            sid_type = 'students'
            
        self.stud_samples = []
        start = time.time()
        result = None
        score = None
        
        sizes_to_scores = {}
        
        # Begin grading, iterating over all sample sizes
        for i in range(len(self.sample_sizes)):
            
            # Obtain the necessary amount of samples
            sample_size = self.sample_sizes[i]
            self.sample(sample_size, sid, pos=pos, show_sample_progress=show_sample_progress)
            
            # Check if we obtained the appropriate amount of samples
            bad_single_dim = len(self.stud_samples) < sample_size
            if 'array' in self.dtype:
                n_dims = tuple([int(i) for i in self.dtype.split('array_shape_')[1][1:-1].split(',') if i != ''])
            else:
                n_dims = 1
            bad_multi_dim = self.stud_samples.shape[1:] != n_dims
            if bad_single_dim or (bad_multi_dim and 'array' in self.dtype):
                # The program is marked as incorrect and given a very discrepant score
                result = False, 1e7, sample_size, time.time() - start, {sample_size: 1e7}
                break
                
            # If multidimensional, we need the projections
            if 'array' in self.dtype: 
                proj_file = 'orthogonal_projections.npy' if self.proj_method == 'OP' else 'euclidean_distances.npy'
                stud_file_path = os.path.join(DATA_DIR, self.qid, sid_type, sid, self.test_label, proj_file)
                soln_file_path = os.path.join(DATA_DIR, self.qid, 'solution', 'solution', self.test_label, proj_file)
                stud_dists = np.load(stud_file_path, allow_pickle=True)
                soln_dists = np.load(soln_file_path, allow_pickle=True)
                np.random.shuffle(stud_dists)
                np.random.shuffle(soln_dists)
            else:
                stud_dists = self.stud_samples
                soln_dists = self.soln_samples
            
            # Determine tolerance for the given FRR
            if len(self.sample_sizes) > 1:
                frr = self.frr / 2 ** (len(self.sample_sizes) - i)
            else:
                frr = self.frr
            score = self.scorer.compute_score(stud_dists[:sample_size], soln_dists[:self.n_soln_samples])
            monte_carlo_path = os.path.join(
                DATA_DIR, self.qid, 'solution', 'mc_solutions', self.test_label, f'mc_scores_{str(self.scorer)}.json'
            )
            epsilon = self.scorer.rejection_threshold(
                frr, sample_size, self.n_soln_samples, monte_carlo_path=monte_carlo_path
            )   
            sizes_to_scores[sample_size] = score

            # Determine if score is above the rejection threshold
            # In either case here, the program is marked as incorrect
            if type(epsilon) == dict:  # Two-sided threshold
                if score <= epsilon['lower_threshold'] or score >= epsilon['upper_threshold']:
                    result = False, score, sample_size, time.time() - start, sizes_to_scores 
                    break
            else:  # Single-sided threshold
                if score >= epsilon: 
                    result = False, score, sample_size, time.time() - start, sizes_to_scores
                    break
       
        # The program is marked as correct
        if result is None:
            result = True, score, self.sample_sizes[-1], time.time() - start, sizes_to_scores
        
        # Remove samples and projections if requested
        if delete_samples:
            sample_file = os.path.join(DATA_DIR, self.qid, sid_type, sid, self.test_label, 'samples.npy')
            os.remove(sample_file)
            if 'array' in dtype:
                proj_file = 'orthogonal_projections.npy' if self.proj_method == 'OP' else 'euclidean_distances.npy'
                sample_file = os.path.join(DATA_DIR, self.qid, sid_type, sid, self.test_label, proj_file)
                os.remove(sample_file)
            
        return result
        
    def grade_wrapper(self, sid, pos, proc_queue, delete_samples, show_sample_progress):
        """
        Wrapper for grading a single student ID. It is called as a subprocess for grade_parallel.
        
        Parameters:
        sid (str):                   the student ID
        pos (int):                   the progress bar position
        proc_queue (Queue):          the process queue
        delete_samples (bool):       whether to keep or discard collected samples
        show_sample_progress (bool): show the sample progress bar
        
        Returns:
        None
        """
        result = self.grade(sid, pos=pos, delete_samples=delete_samples, show_sample_progress=show_sample_progress)
        if proc_queue is not None:
            proc_queue.put((mp.current_process().pid, pos, sid, result))
    
    def grade_parallel(self, sids, max_parallel=20, delete_samples=False, show_sample_progress=True):
        """
        Grade multiple student submissions in parallel. 
        
        Parameters:
        sids (list of str):          the student IDs corresponding to submissions
        max_parallel (int):          the maximum number of parallel processes for grading
        delete_samples (bool):       whether to keep or discard collected samples
        show_sample_progress (bool): show the sample progress bar
        
        Returns: 
        results (dict): a dictionary with student IDs as keys and their corresponding grading results
        """

        results = {}

        # Set up parallelization
        proc_dict = dict()
        proc_queue = mp.Queue(max_parallel)
        filled_pos = 0
        pbar = tqdm(sids, total=len(sids))

        # Handle dequeueing 
        def _dequeue_proc():
            old_pid, old_pos, sid, result = proc_queue.get()
            old_proc = proc_dict[old_pid]
            old_proc.join()
            old_proc.close()
            results[sid] = result  
            pbar.update(1)       
            return old_pos

        # Begin sampling for each sid
        for sid in sids:
            if filled_pos >= max_parallel:
                pos = _dequeue_proc()
            else:
                pos = filled_pos + 1
            filled_pos += 1
            p = mp.Process(target=self.grade_wrapper, 
                           args=(sid, pos, proc_queue, delete_samples, show_sample_progress))
            p.start()
            proc_dict[p.pid] = p

        for _ in range(max_parallel):
            _dequeue_proc()

        return results


if __name__ == '__main__':
    
    print('\n\n- - - - - GRADING - - - - -\n')
    
    # Load in command line arguments
    parser = argparse.ArgumentParser(description='Grades student programs.')
    parser.add_argument('qid', type=str, help='the question ID')
    parser.add_argument('--sids_file_path', type=str, 
                        help='file path to a .txt file of student IDs, with each ID on a separate line')
    parser.add_argument('--max_parallel', type=int, default=10, 
                        help='maximum number of parallel processes for grading')
    parser.add_argument('--delete_samples', action='store_true', 
                        help='delete samples/projections from sample directory for space conservation')
    parser.add_argument('--no_best_n', action='store_true', 
                        help='use the max value of N from config.ini instead of best_n (if choose_n was run)')
    args = parser.parse_args()
    qid = args.qid
    sids_file_path = args.sids_file_path
    
    # Load in parameters found in preprocess.py
    config = configparser.ConfigParser()
    if os.path.isfile(os.path.join(DATA_DIR, qid, 'setup', 'config.ini')):   
        config.read(os.path.join(DATA_DIR, qid, 'setup', 'config.ini'))
    else:
        raise ValueError(f'No config.ini found in the {qid} directory! Did you run preprocess.py?')
        
    print('Loading model parameters.')
    dtype = config['Parameters']['dtype']
    func_name = config['Parameters']['func_name']
    frr = float(config['Parameters']['frr'])
    max_n = int(config['Parameters']['max_n'])
    min_n = int(config['Parameters']['min_n'])
    n_scale_factor = int(config['Parameters']['n_scale_factor'])
    n_soln_samples = int(config['Parameters']['n_soln_samples'])
    scorer_name = config['Parameters']['scorer']
    scorer_map = make_scorer_map()
    scorer = scorer_map[scorer_name]
    if 'proj_method' in config['Parameters'].keys():
        proj_method = config['Parameters']['proj_method']
    else:
        proj_method = None
    
    # Load in the maximum value of n if already obtained from choose_n.py
    if not args.no_best_n:
        best_n_path = os.path.join(DATA_DIR, qid, 'setup', f'best_n_far={frr}.txt')
        if os.path.isfile(best_n_path):
            with open(best_n_path) as f:
                max_n = int(f.read().split('=')[1])
    
    # Determine sample sizes
    sample_sizes = []
    curr = min_n
    while curr < max_n:
        sample_sizes.append(curr)
        curr *= n_scale_factor
    sample_sizes.append(max_n)
    
    # Determine the student IDs to be graded
    print('Gathering student IDs.')
    if sids_file_path:
        with open(sids_file_path) as f:
             sids_to_grade = f.readlines()
    else:
        sids_to_grade = os.listdir(os.path.join(DATA_DIR, qid, 'students'))
        
    # Instatiate StochasticGrade and begin grading
    print(f'Grading {len(sids_to_grade)} students.\n')
    algorithm = StochasticGrade(qid, scorer, sample_sizes, frr, dtype, func_name, n_soln_samples, proj_method=proj_method)
    results = algorithm.grade_parallel(sids_to_grade, max_parallel=args.max_parallel, delete_samples=args.delete_samples)
    
    # Process and save results:
    print('\nProcessing and saving results.')
    predictions, scores, samples_needed, runtimes, sample_sizes_to_scores = {}, {}, {}, {}, {}
    dicts = [predictions, scores, samples_needed, runtimes, sample_sizes_to_scores]
    files = ['predictions', 'final_scores', 'samples_needed', 'runtimes', 'sample_sizes_to_scores']
    
    for sid in results:
        result = results[sid]
        out = {
            'prediction': result[0],
            'final_score': result[1],
            'samples_needed': result[2],
            'runtime': result[3]
        }
        
        with open(os.path.join(DATA_DIR, qid, 'students', sid, 'grading_result.json'), 'w') as f:
            json.dump(out, f, indent=4)
            
        for i in range(5):
            dicts[i][sid] = result[i]
    
    for i in range(5):
        with open(os.path.join(DATA_DIR, qid, 'results', f'{files[i]}.json'), 'w') as f:
            json.dump(dicts[i], f, indent=4)
            
            
    # Generate a summative report 
    print('Generating a summative report.')
    num_students = len(results)
    num_correct = [results[sid][0] for sid in results].count(True)
    out = '- - - - - SUMMATIVE REPORT - - - - -\n\n'
    out += f'Graded {num_students} students on their implementation by executing: {func_name}.\n'
    out += f'Students were graded using scorer: {str(scorer)}.\n\n'
    out += f'{num_correct} students were correct.\n'
    out += f'{num_students - num_correct} students were incorrect.\n'
    out += f'Overall, {num_correct / num_students * 100:.1f}% of students wrote a correct submission.'
    with open(os.path.join(DATA_DIR, qid, 'results', 'report.txt'), 'w') as f:
        f.write(out)
    
    print('Success!\n\n')
            
    
