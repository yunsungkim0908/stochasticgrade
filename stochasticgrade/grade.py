import json
import numpy as np
import os
import time

from stochasticgrade.constants import DATA_DIR
from stochasticgrade.score import *

class StochasticGrade():
    """
    Implements the StochasticGrade algorithm.
    """
    
    def __init__(self, qid, scorer, sample_sizes, false_rejection_rate, dtype, func_name,
                 num_soln_samples=500000, test_label='', test_args=[]):
        self.qid = qid
        self.scorer = scorer
        self.sample_sizes = sample_sizes
        self.frr = false_rejection_rate
        self.dtype = dtype
        self.func_name = func_name
        self.num_soln_samples = num_soln_samples
        self.test_label = test_label
        self.test_args = test_args
        
        soln_sample_path = os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 'samples.npy')
        if os.path.isfile(soln_sample_path):
            self.soln_samples = np.load(soln_sample_path, allow_pickle=True)
        else:
            raise ValueError('No solution samples found. Be sure to execute preprocess.py!')
        np.random.shuffle(self.soln_samples)
        
    def sample(self, n, sid):
        """
        Expand the collection of total student samples to n.
        """      
        sid_type = 'solution' if 'solution' in sid else 'students'
        file_path = os.path.join(DATA_DIR, self.qid, sid_type, sid, self.test_label, 'samples.npy')
        if not os.path.isfile(file_path):
            num_samples = n
        else:
            self.stud_samples = np.load(file_path, allow_pickle=True)
            num_samples = n - len(self.stud_samples)
        
        if num_samples > 0:
            sample_sid_single(
                sid, self.qid, num_samples, self.dtype, self.func_name, 
                test_label=self.test_label, test_args=self.test_args, append_samples=True
            )  
        self.stud_samples = np.load(file_path, allow_pickle=True)
        np.random.shuffle(self.stud_samples)
         
    def grade(self, sid):
        """
        Grade the given student program for the given test suite. 
        """  
        self.stud_samples = []
        start = time.time()
        
        for i in range(len(self.sample_sizes)):
            
            # Obtain the necessary amount of samples
            sample_size = self.sample_sizes[i]
            self.sample(sample_size, sid)
            
            # Check if we obtained the appropriate amount of samples
            bad_single_dim = len(self.stud_samples) < sample_size
            if 'array_shape_' in self.dtype:
                n_dims = tuple([int(i) for i in self.dtype.split('array_shape_')[1][1:-1].split(',')])
            else:
                n_dims = 1
            bad_multi_dim = self.stud_samples.shape[1:] != n_dims
            if bad_single_dim or (bad_multi_dim and 'array_shape_' in self.dtype):
                return False, 1e7, sample_size, time.time() - start
            
            # Determine tolerance for the given FRR
            if len(self.sample_sizes) > 1:
                frr = self.frr / 2 ** (len(self.sample_sizes) - i)
            else:
                frr = self.frr
            score = self.scorer.compute_score(
                self.stud_samples[:sample_size], self.soln_samples[:self.num_soln_samples], sid, self.qid
            )
            monte_carlo_path = os.path.join(
                DATA_DIR, self.qid, 'results', self.test_label, str(self.scorer), 'monte_carlo_scores.json'
            )
            epsilon = self.scorer.rejection_threshold(
                frr, sample_size, self.num_soln_samples, monte_carlo_path=monte_carlo_path
            )   

            # Determine if score is above the rejection threshold
            if type(epsilon) == dict:  # Two-sided threshold
                if score <= epsilon['lower_threshold'] or score >= epsilon['upper_threshold']:
                    return False, score, sample_size, time.time() - start  
            else:  # Single-sided threshold
                if score >= epsilon: 
                    return False, score, sample_size, time.time() - start

        return True, score, sample_size, time.time() - start