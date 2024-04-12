import argparse
import configparser
import json
import math
import multiprocessing
import numpy as np
import os
import pickle
import re
import sklearn.metrics
import sklearn.preprocessing
import shutil
import subprocess
import time

from scipy import stats
from tqdm import tqdm

from stochasticgrade.constants import *
from stochasticgrade.sample import *
from stochasticgrade.score import *
from stochasticgrade.utils import * 


class StochasticGrade():
    """
    Implements the StochasticGrade algorithm.
    """
    
    def __init__(self, qid, scorer, sample_sizes, false_rejection_rate, dtype, num_soln_samples=500000, 
                 test_label='', test_args=[], proj_method='ED', dilation=1):
        """
        Initializes a new instance of StochasticGrade and collects solution samples.
        
        Parameters:
        qid (str):                    the question ID for the problem to be evaluated
        scorer (Scorer):              the scorer to be used for evaluation
        sample_sizes (list of int):   the sample size schedule
        false_rejection_rate (float): the false rejection rate for grading
        dtype (str):                  the data type of the student samples
        num_soln_samples (int):       the number of solution samples to collect
        test_label (str):             the name of the test case to be evaluated
        test_args (list):             the arguments to be used for the test case
        proj_method (str):            the projection method for multivariate grading
        dilation (int):               the dilation factor for the bounding box random anchor point
        
        Returns:
        None
        """
        self.qid = qid
        self.scorer = scorer
        self.sample_sizes = sample_sizes
        self.frr = false_rejection_rate
        self.dtype = dtype
        self.num_soln_samples = num_soln_samples
        self.test_label = test_label
        self.test_args = test_args
        self.proj_method = proj_method
        self.dilation = dilation
        
        soln_sample_path = os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 'samples.npy')
        if os.path.isfile(soln_sample_path):
            self.soln_samples = np.load(soln_sample_path, allow_pickle=True)
        else:
            self.sample_solution()
        np.random.shuffle(self.soln_samples)
        
    def sample_solution(self):
        """
        Sample `num_soln_samples` from the solution program, stored in `soln_samples`.
        
        Parameters:
        None
        
        Returns: 
        None
        """
        print('Sampling the solution program.')
        get_student_info_single(
            'solution', self.qid, self.num_soln_samples, self.dtype, 
            test_label=self.test_label, test_args=self.test_args, append_samples=False
        )  
        file_path = os.path.join(DATA_DIR, self.qid, 'solution', 'solution', self.test_label, 'samples.npy')
        self.soln_samples = np.load(file_path, allow_pickle=True)
        print(f'Generated {self.num_soln_samples} solution samples.')
        
    def sample(self, n, sid):
        """
        Expand the collection of total student samples to size n, stored in `stud_samples`.
        
        Parameters: 
        n (int): the total number of samples to expand to
        sid (str): the student ID for the student whose program is being sampled
        
        Returns: 
        None
        """      
        sid_type = 'solution' if 'solution' in sid else 'students'
        file_path = os.path.join(DATA_DIR, self.qid, sid_type, sid, self.test_label, 'samples.npy')
        if not os.path.isfile(file_path):
            num_samples = n
        else:
            self.stud_samples = np.load(file_path, allow_pickle=True)
            num_samples = n - len(self.stud_samples)
        
        if num_samples > 0:
            get_student_info_single(
                sid, self.qid, num_samples, self.dtype, 
                test_label=self.test_label, test_args=self.test_args, append_samples=True
            )  
        self.stud_samples = np.load(file_path, allow_pickle=True)
        np.random.shuffle(self.stud_samples)
        
    def monte_carlo(self, num_samples, M=1000, max_parallel=20, overwrite=False):
        """
        Sample M solution program sample sets of size `num_samples`.
        
        Parameters: 
        num_samples (int):  the number of samples within each sample set
        M (int):            the number of sample sets to create
        max_parallel (int): the maximum number of parallel processes for sampling
        overwrite (bool):   overwrite existing Monte Carlo samples
        
        Returns:
        None
        """
        monte_carlo_dir = os.path.join(DATA_DIR, self.qid, 'solution', 'mc_solutions', self.test_label)        
        if not os.path.isdir(monte_carlo_dir):
            os.makedirs(monte_carlo_dir)
                
        sids = [f'mc_solution_{i+1}' for i in range(M)]
        sample_sids = []
        for sid in sids:
            if sid not in os.listdir(monte_carlo_dir) or overwrite:
                sample_sids.append(sid)
        if sample_sids:
            get_student_info_multi(
                sample_sids, self.qid, num_samples, self.dtype, max_parallel=max_parallel,
                test_label=self.test_label, test_args=self.test_args, append_samples=True
            )
         
    def grade(self, sid):
        """
        Grade the given student program for the given test suite (if it exists). 
        
        Parameters:
        sid (str): the student ID for the student whose program is being graded
        
        Returns:
        A tuple with contents
        - [0]: prediction of whether the program was correct (True) or not (False)
        - [1]: the final evaluation score
        - [2]: the sample size at which the final prediction occurred
        - [3]: the total runtime for grading the program
        - [4]: the scores obtained at each sample size
        """  
        self.stud_samples = []
        all_scores = {}  # Keep track of scores calculated at each step
        start = time.time()
        
        for i in range(len(self.sample_sizes)):
            
            # Obtain the necessary amount of samples according to the sample size schedule
            sample_size = self.sample_sizes[i]
            self.sample(sample_size, sid)
            
            # Check if we obtained the appropriate amount of samples
            if 'array_shape' in self.dtype:
                n_dims = tuple([int(i) for i in self.dtype.split('array_shape_')[1][1:-1].split(',')])
            else:
                n_dims = 1
            bad_single_dim = len(self.stud_samples) < sample_size
            bad_multi_dim = self.stud_samples.shape[1:] != n_dims
            improper_shape = bad_single_dim or (bad_multi_dim and 'array_shape' in self.dtype)
            if improper_shape:
                return False, 1e7, sample_size, time.time() - start, {sample_size: 1e7}
            
            # Use samples directly in the case of single-dimensional samples, otherwise
            # use the Euclidean distance projection (ED) or orthogonal projection (OP)
            stud_samples = self.stud_samples[:sample_size]
            soln_samples = self.soln_samples[:self.num_soln_samples]
            if 'array_shape' in self.dtype:
                if self.proj_method == 'ED':  # Euclidean Distance
                    stud_dists = get_euclidean_distances(stud_samples, soln_samples, sid, self.qid, dilation=self.dilation)
                    soln_dists = get_euclidean_distances(soln_samples, soln_samples, sid, self.qid, dilation=self.dilation)
                    np.save(os.path.join(DATA_DIR, qid, sid_type, sid, test_label, 
                                         'euclidean_distances.npy'), stud_dists)
                    np.save(os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 
                                         'euclidean_distances.npy'), soln_dists)
                else:  # Orthogonal Projection
                    stud_dists = get_orthogonal_projection(stud_samples, soln_samples, sid, self.qid, dilation=self.dilation)
                    soln_dists = get_orthogonal_projection(soln_samples, soln_samples, sid, self.qid, dilation=self.dilation)
                    np.save(os.path.join(DATA_DIR, qid, sid_type, sid, test_label, 
                                         'orthogonal_projection.npy'), stud_dists)
                    np.save(os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 
                                         'orthogonal_projection.npy'), soln_dists)
            else:
                stud_dists = stud_samples
                soln_dists = soln_samples
            
            # Determine tolerance for the given FRR
            frr = self.frr / 2 ** (len(self.sample_sizes) - i)
            score = self.scorer.compute_score(stud_dists, soln_dists, sid, self.qid)
            all_scores[sample_size] = score
             scorer_path = str(self.scorer)+self.proj_method if 'array_shape' in self.dtype else str(self.scorer)
            monte_carlo_path = os.path.join(
                DATA_DIR, self.qid, 'results', self.test_label, scorer_path, proj_path, 'monte_carlo_scores.json'
            )
            epsilon = self.scorer.rejection_threshold(
                frr, sample_size, self.num_soln_samples, monte_carlo_path=monte_carlo_path
            )   

            # Determine if score is above the rejection threshold
            if type(epsilon) == dict:  # Two-sided threshold
                if score <= epsilon['lower_threshold'] or score >= epsilon['upper_threshold']:
                    return False, score, sample_size, time.time() - start, all_scores 
            else:  # Single-sided threshold
                if score >= epsilon: 
                    return False, score, sample_size, time.time() - start, all_scores

        return True, score, sample_size, time.time() - start, all_scores
                    
            
if __name__ == '__main__':
    # Set up command line arguments
    """
    The arguments are used as follows:
    
    Required: 
    qid (str): the question ID for the problem to be evaluated
    
    Optional: 
    data-file (str):     the path to a data file (.json) containing sid (str): program (str) mappings. If not
                         provided, then all programs within the existing `students` directory for the problem 
                         are evaluated
    closest_error (str): the path to a text file (.txt) containing the function for the nearest error that
                         obeys the FRR bound.
    sid (str):           a student ID to be evaluated in isolation, instead of evaluating students as a batch
    mode (str):          the action to be performed (e.g. grading, sampling, clustering). Defaults to batch-grade
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('qid', type=str)
    parser.add_argument('--data-file')
    parser.add_argument('--closest-error', type=str)
    parser.add_argument('--sid', type=str)
    parser.add_argument('--mode', type=str, 
                        choices=['single-grade', 'batch-grade', 'cluster', 'monte-carlo', 'hand-label',
                                'single-sample', 'batch-sample', 'choose_n'], default='batch-grade')
    args = parser.parse_args()
    
    # Load the command line arguments
    single_grade_sid = args.sid
    qid = args.qid
    mode = args.mode
    
    # Load the config file
    config = configparser.ConfigParser()
    if os.path.isfile(os.path.join(DATA_DIR, qid, 'config.ini')):   
        config.read(os.path.join(DATA_DIR, qid, 'config.ini'))
    else:
        print('Please create a config.ini file in the qid directory.')
        raise Exception
            
    # Load the model parameters from the config file
    dilation = config['Multivariate Parameters']['dilation']
    dtype = config['Parameters']['dtype']
    frr = float(config['Parameters']['false_rejection_rate'])
    max_parallel = int(config['Parameters']['max_parallel'])
    num_soln_samples = int(config['Parameters']['num_soln_samples'])
    proj_method = config['Multivariate Parameters']['projection_method']
    sample_sizes = [int(num) for num in config['Parameters']['sample_sizes'].split(',')]
    scorer_name = config['Parameters']['scorer']
    scorer_map = make_scorer_map()
    scorer = scorer_map[scorer_name]
    
    # Load the test suite parameters
    all_test_suites = config['Test Suites']['all_test_suites'] == 'True'
    num_test_suites = int(config['Test Suites']['num_test_suites'])
    if num_test_suites > 0:
        test_suite_cases = [f'case_{int(num)}' for num in config['Test Suites']['test_suite_cases'].split(',')]
        test_suite_dir = os.path.join(DATA_DIR, qid, 'test_suites')
        if not os.path.isdir(test_suite_dir):
            os.mkdir(test_suite_dir)
        with open(os.path.join(test_suite_dir, f'{qid}.labels.json'), 'w') as f:
            json.dump(test_suite_cases, f)
        
        test_suites = get_test_suite(
            qid, num_tests=num_test_suites, all_test_suites=all_test_suites
        )
    else:
        test_suites = {'': []}
    
    # Load the clustering parameters
    bandwidth = float(config['Clustering']['bandwidth'])
    cluster_method = config['Clustering']['cluster_method']
    cluster_samples = [int(num) for num in config['Clustering']['cluster_samples'].split(',')]
    cluster_scorers = [scorer.strip() for scorer in config['Clustering']['cluster_scorers'].split(',')]
    n_clusters = int(config['Clustering']['n_clusters'])
    
    # Load the Monte Carlo sampling parameters
    M = int(config['Monte Carlo']['M'])
    mc_num_anchor_samples = int(config['Monte Carlo']['mc_num_anchor_samples'])
    mc_num_samples = int(config['Monte Carlo']['mc_num_samples'])
    mc_special_sizes = [int(num) for num in config['Monte Carlo']['mc_special_sizes'].split(',')]
    
    # Load the overwrite parameters
    overwrite_error_samples = True if config['Overwrite']['overwrite_error_samples'].lower() == 'true' else False
    overwrite_labels = True if config['Overwrite']['overwrite_labels'].lower() == 'true' else False
    overwrite_mc_samples = True if config['Overwrite']['overwrite_mc_samples'].lower() == 'true' else False
    overwrite_mc_scores = True if config['Overwrite']['overwrite_mc_scores'].lower() == 'true' else False
    
    # Load in the data for sampling
    if args.data_file:
        print('Loading data...')
        with open(args.data_file) as f:
            data = json.load(f)
        sids = []
        for sid in tqdm(data):
            sids.append(sid)
            sid_type = 'solution' if 'solution' in sid else 'students'
            sid_dir = os.path.join(DATA_DIR, qid, sid_type, sid)
            if not os.path.isdir(sid_dir):
                os.makedirs(sid_dir)
            with open(os.path.join(sid_dir, 'response.txt'), 'w') as f:
                f.write(data[sid])
    else:
        sids = os.listdir(os.path.join(DATA_DIR, qid, 'students'))
        
    # Load the nearest error program, if it exists
    if args.closest_error:
        closest_error_prog = f.read(args.closest_error)
        
    # Create the results directory to store evaluations
    if not os.path.isdir(os.path.join(DATA_DIR, qid, 'results')):
        os.makedirs(os.path.join(DATA_DIR, qid, 'results'))
        
        
    if mode == 'single-grade' or mode == 'batch-grade':
        """
        single-grade: grades a single program specified by an input sid.
        batch-grade:  grades all programs within a given sid: program mapping, or
                      grades all programs within the `students` directory 
        """
        sids_to_grade = [single_grade_sid] if mode == 'single-grade' else sids
        
        # Iterate through all test cases
        for test_label, test_args in test_suites.items():
        
            # Set path for saving results
            scorer_path = scorer_name+proj_method if 'array_shape' in dtype else scorer_name
            results_path = os.path.join(DATA_DIR, qid, 'results', test_label, scorer_path, str(frr))
            if not os.path.isdir(results_path):
                os.makedirs(results_path)

            # Load existing data, if there is any
            file_names = ['labels.json', 'final_scores.json', 'samples_needed.json', 'runtimes.json', 'all_scores.json']
            dicts = [{}, {}, {}, {}, {}]
            for i, name in enumerate(file_names):
                if os.path.isfile(os.path.join(results_path, name)):  
                    with open(os.path.join(results_path, name)) as f:
                        dicts[i] = json.load(f)
            labels, scores, samples_needed, runtimes, all_scores = dicts[0], dicts[1], dicts[2], dicts[3], dicts[4]
            
            # Instantiate the algorithm for grading
            algorithm = StochasticGrade(
                qid, scorer, sample_sizes, frr, dtype, num_soln_samples=num_soln_samples, 
                test_label=test_label, test_args=test_args, proj_method=proj_method
            )

            for i in tqdm(range(len(sids_to_grade))):
                sid = sids_to_grade[i]
                out = algorithm.grade(sid)

                # Save data
                for j, item in enumerate(out):
                    dicts[j][sid] = item
                if i % 5 == 0 or i == len(sids_to_grade) - 1:
                    for j, name in enumerate(file_names):
                        with open(os.path.join(results_path, name), 'w') as f:
                            json.dump(dicts[j], f)
                            
        
    if mode == 'single-sample' or mode == 'batch-sample':
        """
        single-sample: sample `num_soln_samples` times from the given sid
        batch-sample:  samples all programs within a given sid: program mapping, or
                       samples all programs within the `students` directory 
        """
        sids_to_sample = []
        print('Checking for existing samples...')
        for sid in tqdm(sids):
            for test_label, test_args in test_suites.items():
                sid_type = 'solution' if 'solution' in sid else 'students'
                if os.path.isfile(os.path.join(DATA_DIR, qid, sid_type, sid, test_label, 'samples.npy')):
                    samples = np.load(os.path.join(DATA_DIR, qid, sid_type, sid, test_label, 'samples.npy'), allow_pickle=True)
                    if len(samples) == sample_sizes[-1]:
                        continue
                sids_to_sample.append(sid)
                break
                
        sids_to_sample = [single_grade_sid] if mode == 'single-grade' else sids_to_sample
        
        # Iterate through all test cases
        for test_label, test_args in test_suites.items():
        
            # Instantiate the algorithm for sampling
            algorithm = StochasticGrade(
                qid, scorer, sample_sizes, frr, dtype, num_soln_samples=num_soln_samples, 
                test_label=test_label, test_args=test_args, proj_method=proj_method
            )
            
            # Execute parallelized sampling
            get_student_info_multi(
                sids_to_sample, qid, max(sample_sizes), dtype, max_parallel=max_parallel,
                test_label=test_label, test_args=test_args, append_samples=True
            )
    
    
    if mode == 'cluster':
        """
        Cluster programs based on their scores using the clustering method and parameters
        specified by the config file. 
        """
        for sid in sids:
            for test_label, _ in test_suites.items():
                sid_type = 'solution' if 'solution' in sid else 'students'
                if not os.path.isfile(os.path.join(DATA_DIR, qid, sid_type, sid, test_label, 'samples.npy')):
                    print('Must have samples for all student IDs that will be clustered!')
                    raise Exception
        
        for num_cluster_samples in cluster_samples:
            score_list = []
            print(f'Clustering using {num_cluster_samples} samples. Creating {n_clusters} clusters.')
            soln_samples = np.load(
                os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 'samples.npy'), allow_pickle=True
            )
            for scorer in cluster_scorers:
                scorer = scorer_map[scorer]
                
                print(f'Computing scores under {str(scorer)}...')
                for test_label, _ in test_suites.items():
                    scores = []
                    for sid in tqdm(sids):
                        if 'solution' in sid:
                            sid_type = 'solution':
                        elif 'closest_error' in sid:
                            sid_type = 'closest_error'
                        else:
                            sid_type = 'students'
                        stud_samples = np.load(
                            os.path.join(DATA_DIR, qid, sid_type, sid, test_label, 'samples.npy'), allow_pickle=True
                        )
                        
                        # Check if the score already exists for the desired sample size
                        frr = os.listdir(os.path.join(DATA_DIR, qid, sid_type, 'results', str(scorer)))[0] 
                        scorer_path = str(scorer)+proj_method if 'array_shape' in dtype else str(scorer)
                        score_path = os.path.join(
                            DATA_DIR, qid, sid_type, 'results', scorer_path, frr, 'all_scores.json'
                        )
                        if os.path.isfile(score_path):
                            with open(score_path) as f:
                                existing_scores = json.load(f)
                            score = existing_scores[sid][num_cluster_samples]
                            scores.append(score)
                        continue
                        
                        # If not, we calculate the score.
                        # Use samples directly in the case of single-dimensional samples, otherwise
                        # use the Euclidean distance projection (ED) or orthogonal projection (OP)
                        s1 = stud_samples[:num_cluster_samples]
                        s2 = soln_samples[:num_soln_samples]
                        if 'array_shape' in dtype:
                            if proj_method == 'ED':  # Euclidean Distance
                                stud_dists = get_euclidean_distances(s1, s2, sid, qid, dilation=dilation)
                                soln_dists = get_euclidean_distances(s2, s2, sid, qid, dilation=dilation)
                                np.save(os.path.join(DATA_DIR, qid, sid_type, sid, test_label, 
                                                     'euclidean_distances.npy'), stud_dists)
                                np.save(os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 
                                                     'euclidean_distances.npy'), soln_dists)
                            else:  # Orthogonal Projection
                                stud_dists = get_orthogonal_projection(s1, s2, sid, qid, dilation=dilation)
                                soln_dists = get_orthogonal_projection(s2, s2, sid, qid, dilation=dilation)
                                np.save(os.path.join(DATA_DIR, qid, sid_type, sid, test_label, 
                                                     'orthogonal_projection.npy'), stud_dists)
                                np.save(os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 
                                                     'orthogonal_projection.npy'), soln_dists)
                        else:
                            stud_dists = s1
                            soln_dists = s2
                            
                        # Check if we've obtained the appropriate amount of samples
                        if 'array_shape' in dtype:
                            n_dims = tuple([int(i) for i in dtype.split('array_shape_')[1][1:-1].split(',')])
                        else:
                            n_dims = 1
                        bad_single_dim = len(stud_samples) < num_cluster_samples
                        bad_multi_dim = stud_samples.shape[1:] != n_dims
                        improper_shape = bad_single_dim or (bad_multi_dim and 'array_shape_' in dtype)
                        if improper_shape:
                            score = 1e7
                        else:
                            score = scorer.compute_score(stud_dists, soln_dists, sid, qid)
                        scores.append(score)
                    score_list.append(scores)

            if cluster_method == 'agglomerative':
                cluster_labels = agglomerative(score_list, n_clusters=n_clusters)    
                n_clusters_path = str(n_clusters)
            elif cluster_method == 'mean_shift':
                cluster_labels = mean_shift(score_list, bandwidth=bandwidth)
                n_clusters_path = str(bandwidth)
            else:
                print('Please select a valid clustering method: `agglomerative` or `mean_shift`.')
                raise Exception

            cluster_path = os.path.join(DATA_DIR, qid, 'results', 'clusters')
            scorer_names = ''
            for scorer in cluster_scorers:
                scorer_names += scorer + '+' + proj_method if 'array_shape' in dtype else scorer + '+' 
            scorer_names = scorer_names[:-1]
            if num_test_suites > 0:
                path = os.path.join(cluster_path, scorer_names, f'ncases={str(len(test_suite_cases))}',
                                    str(num_cluster_samples))
            else:
                path = os.path.join(cluster_path, scorer_names, str(num_cluster_samples))
            if not os.path.isdir(path):
                os.makedirs(path)
                
            cluster_labels = {sids[i]: int(cluster_labels[i]) for i in range(len(sids))}
            score_list = {sids[i]: list(np.array(score_list)[:, i]) for i in range(len(sids))}  
            
            with open(os.path.join(path, 'clusters.json'), 'w') as f:
                json.dump(cluster_labels, f)
            with open(os.path.join(path, 'scores.json'), 'w') as f:
                json.dump(score_list, f)
            

    if mode == 'hand-label':
        """
        Label student programs by hand to establish the ground truth.
        Running this mode will enable you to follow the instructions from an 
        on-screen prompt for labeling. 
        """
        
        # Load existing data, if there is any
        gt_labels, gt = {}, {}
        if os.path.isfile(os.path.join(DATA_DIR, qid, 'results', 'ground_truth_labels.json')):
            with open(os.path.join(DATA_DIR, qid, 'results', 'ground_truth_labels.json')) as f:
                gt_labels = json.load(f)
        if os.path.isfile(os.path.join(DATA_DIR, qid, 'results', 'ground_truth.json')):
            with open(os.path.join(DATA_DIR, qid, 'results', 'ground_truth.json')) as f:
                gt = json.load(f)
        
        # Label the given sids
        for i in range(len(sids)):
            sid = sids[i]
            if sid not in gt_labels or sid not in gt or overwrite_labels:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f'Program:  {i+1} / {len(sids)} \t ({format((i+1)/len(sids) * 100, ".1f")}% complete)')
                print('- - - - - - - - - - - - -\n')
                print(f'Student: {sid}\n\n')
                sid_type = 'solution' if 'solution' in sid else 'students'
                with open(os.path.join(DATA_DIR, qid, sid_type, sid, 'response.txt')) as f:
                    prog = f.read()
                print(prog)
                print('\n\n')
                print('Label format: {C/I}, Tag#1, Tag#2, ..., Tag#n')
                print('Choose either C (correct) or I (incorrect)')
                print('Press ENTER to skip\n')
                label = input('Enter a label: ')
                if label == '':
                    continue
                gt_labels[sid] = label
                gt[sid] = label.split(',')[0]
                with open(os.path.join(DATA_DIR, qid, 'results', 'ground_truth_labels.json'), 'w') as f:
                    json.dump(gt_labels, f)
                with open(os.path.join(DATA_DIR, qid, 'results', 'ground_truth.json'), 'w') as f:
                    json.dump(gt, f)
        
        
    if mode == 'monte-carlo':
        """
        Perform Monte Carlo sampling and precompute scores for the provided rejection threshold.
        """
        algorithm = StochasticGrade(qid, scorer, sample_sizes, frr, dtype, num_soln_samples)

        # Obtain the current number of Monte Carlo Samples to see if we need more
        regex = r'mc_solution_(\d+)'
        max_num = -1
        for filename in os.listdir(os.path.join(DATA_DIR, qid, 'solution', 'mc_solutions')):
            match = re.search(regex, filename)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
        
        if overwrite_mc_samples or max_num < M:
            num_samples = mc_num_anchor_samples if mc_num_anchor_samples > max(sample_sizes) else max(sample_sizes)
            print(f'Sampling {num_samples} samples from the correct distribution...')
            algorithm.monte_carlo(num_samples, M=M, max_parallel=max_parallel, overwrite=overwrite_mc_samples)
                        
        scorer_path = scorer_name+algorithm.proj_method if 'array_shape' in self.dtype else scorer_name
        results_path = os.path.join(DATA_DIR, qid, 'results', scorer_path)
        scores_path = os.path.join(results_path, 'monte_carlo_scores.json')
        
        # Precompute scores for Monte Carlo samples
        if os.path.isfile(scores_path) and not overwrite_mc_scores:
            with open(scores_path) as f:
                scores = json.load(f)
        else:
            scores = {}
            
        monte_carlo_dir = os.path.join(DATA_DIR, qid, 'solution', 'mc_solutions')
        print('Computing scores...')
        
        for sample_size in sample_sizes + [mc_num_samples] + mc_special_sizes:
            if str(sample_size) in scores:
                continue
            scores[sample_size] = []
            for sid in tqdm(os.listdir(monte_carlo_dir)):
                samples = np.load(os.path.join(monte_carlo_dir, sid, 'samples.npy'), allow_pickle=True)
                np.random.shuffle(samples)
                
                # Use samples directly in the case of single-dimensional samples, otherwise
                # use the Euclidean distance projection (ED) or orthogonal projection (OP)
                s1 = samples[:sample_size]
                s2 = algorithm.soln_samples[:mc_num_anchor_samples]
                if 'array_shape' in dtype:
                    if proj_method == 'ED':  # Euclidean Distance
                        stud_dists = get_euclidean_distances(s1, s2, sid, qid, dilation=dilation)
                        soln_dists = get_euclidean_distances(s2, s2, sid, qid, dilation=dilation)
                        np.save(os.path.join(DATA_DIR, qid, sid_type, sid, test_label, 
                                             'euclidean_distances.npy'), stud_dists)
                        np.save(os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 
                                             'euclidean_distances.npy'), soln_dists)
                    else:  # Orthogonal Projection
                        stud_dists = get_orthogonal_projection(s1, s2, sid, qid, dilation=dilation)
                        soln_dists = get_orthogonal_projection(s2, s2, sid, qid, dilation=dilation)
                        np.save(os.path.join(DATA_DIR, qid, sid_type, sid, test_label, 
                                             'orthogonal_projection.npy'), stud_dists)
                        np.save(os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 
                                             'orthogonal_projection.npy'), soln_dists)
                else:
                    stud_dists = s1
                    soln_dists = s2
                
                score = algorithm.scorer.compute_score(stud_dists, soln_dists, sid, qid)
                if np.isfinite(score):
                    scores[sample_size].append(score)
                
        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        with open(scores_path, 'w') as f:
            json.dump(scores, f)            
            
            
    if mode == 'choose_n':
        """
        Computes the number of samples n needed to successfully detect FAR fraction
        of the error specified in the error file. 
        """
        # TODO!
        
        # Description: The instructor passes in a string denoting the closest error that they’d like to be able to catch reliably. We read this program and then generate 1,000 samples for this error type. We then compute the maximum number of samples we need in order to catch the guaranteed FRR’s worth of errors.
        
        algorithm = StochasticGrade(qid, scorer, sample_sizes, frr, dtype, num_soln_samples)
        # threshold = algorithm.scorer.rejection_threshold(frr, num_stud_samples, num_soln_samples, monte_carlo_path=None)
        
        closest_error_dir = os.path.join(DATA_DIR, qid, 'closest_error')        
        if not os.path.isdir(closest_error_dir):
            os.makedirs(closest_error_dir)
                
        sids = [f'closest_error_{i+1}' for i in range(M)]
        sample_sids = []
        for sid in sids:
            if sid not in os.listdir(closest_error_dir) or overwrite_error_samples:
                sample_sids.append(sid)
        
        # Generate 1,000 samples for the given error type
        if sample_sids:
            get_student_info_multi(
                sample_sids, qid, num_samples, dtype, max_parallel=max_parallel, append_samples=True
            )
            
        # Compute scores for the samples 
        scorer_path = str(scorer)+proj_method if 'array_shape' in self.dtype else str(scorer)
        results_path = os.path.join(DATA_DIR, qid, 'results', scorer_path)
        scores_path = os.path.join(results_path, 'closest_error_scores.json')
        if os.path.isfile(scores_path):
            with open(scores_path) as f:
                scores = json.load(f)
        else:
            scores = {}
            
        closest_error_dir = os.path.join(DATA_DIR, qid, 'closest_error')
        print('Computing scores for the error programs...')
        
        sample_size = 200
        while sample_size < 5e6:
            if str(sample_size) in scores:
                continue
            scores[sample_size] = []
            for sid in tqdm(os.listdir(closest_error_dir)):
                samples = np.load(os.path.join(closest_error_dir, sid, 'samples.npy'), allow_pickle=True)
                np.random.shuffle(samples)
                
                # Use samples directly in the case of single-dimensional samples, otherwise
                # use the Euclidean distance projection (ED) or orthogonal projection (OP)
                s1 = samples[:sample_size]
                s2 = algorithm.soln_samples[:mc_num_anchor_samples]
                if 'array_shape' in dtype:
                    if proj_method == 'ED':  # Euclidean Distance
                        stud_dists = get_euclidean_distances(s1, s2, sid, qid, dilation=dilation)
                        soln_dists = get_euclidean_distances(s2, s2, sid, qid, dilation=dilation)
                        np.save(os.path.join(DATA_DIR, qid, sid_type, sid, test_label, 
                                             'euclidean_distances.npy'), stud_dists)
                        np.save(os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 
                                             'euclidean_distances.npy'), soln_dists)
                    else:  # Orthogonal Projection
                        stud_dists = get_orthogonal_projection(s1, s2, sid, qid, dilation=dilation)
                        soln_dists = get_orthogonal_projection(s2, s2, sid, qid, dilation=dilation)
                        np.save(os.path.join(DATA_DIR, qid, sid_type, sid, test_label, 
                                             'orthogonal_projection.npy'), stud_dists)
                        np.save(os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 
                                             'orthogonal_projection.npy'), soln_dists)
                else:
                    stud_dists = s1
                    soln_dists = s2
                
                score = algorithm.scorer.compute_score(stud_dists, soln_dists, sid, qid)
                if np.isfinite(score):
                    scores[sample_size].append(score)
                    
            sample_size = min(sample_size * 2, 5e6)
                
        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        with open(scores_path, 'w') as f:
            json.dump(scores, f)            
            
        # Select the value of n that achieves the desired FAR
        