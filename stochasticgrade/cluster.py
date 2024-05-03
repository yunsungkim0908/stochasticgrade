import argparse
import configparser
import json
import numpy as np
import os
import sklearn.cluster
import sklearn.metrics
import sklearn.mixture

from tqdm import tqdm

from stochasticgrade.constants import DATA_DIR
from stochasticgrade.sample import sample_sid_multi
from stochasticgrade.score import *


def agglomerative(score_list, n_clusters=20):    
    """
    Performs agglomerative hierarchical clustering, creating `n_clusters` total.

    Parameters:
    score_list (list of list): a list of lists of scores for various scorers
    n_clusters (int):          the number of clusters to form

    Returns:
    cluster_labels (list): the cluster labels.
    """
    scores = np.column_stack(score_list)
    algorithm = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = algorithm.fit_predict(scores)
    return cluster_labels


def cluster(sids, qid, scorers, sizes, dtype, func_name, n_soln_samples, 
            n_clusters, max_parallel=20, proj_method='ED', test_suites={'': []}):
    """
    Perform agglomerative hierarchical clustering based on scores calculated 
    for each of the student programs. Multiple scorers or input arguments can be
    used as different profiles for a student program, potentially yielding
    better clustering.

    Clusters and scores are saved in the results directory.

    Parameters:
    sids (list of str):       the list of student IDs to be clusters
    qid (str):                the question ID
    scorers (list of Scorer): the list of scorers to be used in clustering
    sizes (list of int):      the list of student program sample sizes to use for scoring
    dtype (str):              the data type of the student program output
    func_name (str):          the function in the student program from which samples are generated
    n_soln_samples (int):     the number of solution samples used for scoring
    n_clusters (int):         the number of clusters to be created
    max_parallel (int):       the maximum number of parallel process for sampling
    proj_method (str):        the projection method used for multidimensional samples
    test_suites (dict):       the dictionary of test labels to test arguments

    Returns:
    None
    """
    
    print('\n\n- - - - - CLUSTERING - - - - -\n')
    
    # Check if samples for the student ID already exist
    # Sample if more samples are needed (sample to the maximum size)
    print('Obtaining the necessary amount of samples.')
    for test_label, test_args in test_suites.items():
        sample_sid_multi(
            sids, qid, max(sizes), dtype, func_name, max_parallel=max_parallel, 
            append_samples=True, test_label=test_label, test_args=test_args, 
            proj_method=proj_method, sample_to=max(sizes)
        )  
    
    # Calculate scores to be used in clustering
    scorer_map = make_scorer_map()
    
    for size in sizes:
        score_list = []
        print(f'\nCalculating scores using {size} samples.')
        
        for scorer in scorers:
            scorer = scorer_map[scorer]
            print(f'- Computing scores under {str(scorer)}.')
            
            for test_label, _ in test_suites.items():
                scores = []
                 
                # Load in solution samples or projections
                if 'array' not in dtype:
                    soln_samples = np.load(os.path.join(DATA_DIR, qid, 'solution', 'solution', 
                                                        test_label, 'samples.npy'), allow_pickle=True)
                else:
                    proj_part = 'orthogonal_projections.npy' if proj_method == 'OP' else 'euclidean_distances.npy'
                    soln_samples = np.load(os.path.join(DATA_DIR, qid, 'solution', 'solution', 
                                                        test_label, proj_part), allow_pickle=True)
                for sid in tqdm(sids):
                    # Load in the student samples
                    if 'solution' in sid:
                        sid_type = 'solution'
                    elif 'closest_error' in sid:
                        sid_type = 'setup'
                    else:
                        sid_type = 'students'
                    stud_samples = np.load(os.path.join(DATA_DIR, qid, sid_type, sid, test_label, 
                                                        'samples.npy'), allow_pickle=True)
                    
                    # Check if we obtained the appropriate amount of samples
                    bad_single_dim = len(stud_samples) < size
                    if 'array' in dtype:
                        n_dims = tuple([int(i) for i in dtype.split('array_shape_')[1][1:-1].split(',')])
                    else:
                        n_dims = 1
                    bad_multi_dim = stud_samples.shape[1:] != n_dims

                    # Load in projections if needed
                    if 'array' in dtype:
                        proj_part = 'orthogonal_projections.npy' if proj_method == 'OP' else 'euclidean_distances.npy'
                        stud_samples = np.load(os.path.join(DATA_DIR, qid, sid_type, sid, 
                                                            test_label, proj_part), allow_pickle=True)

                    # Compute score
                    if bad_single_dim or (bad_multi_dim and 'array' in dtype):
                        score = 1e7
                    else:
                        score = scorer.compute_score(stud_samples[:size], soln_samples[:n_soln_samples])
                    scores.append(score)
                score_list.append(scores)
                
        # Perform agglomerative clustering
        print('\nPerforming agglomerative hierarchical clustering.')
        cluster_labels = agglomerative(score_list, n_clusters=n_clusters)
        
        # Save clustering results
        print('Save clustering results.')
        cluster_path = os.path.join(DATA_DIR, qid, 'results', 'clusters')
        scorer_names = ''
        for scorer in scorers:
            scorer_names += scorer + '+'
        scorer_names = scorer_names[:-1]
        
        test_path = f'ncases={str(len(test_suites))}' if test_suites != {'': []} else ''
        path = os.path.join(cluster_path, scorer_names, test_path, str(size))
        if not os.path.isdir(path):
            os.makedirs(path)

        cluster_labels = {sids[i]: int(cluster_labels[i]) for i in range(len(sids))}
        score_list = {sids[i]: list(np.array(score_list)[:, i]) for i in range(len(sids))}  

        with open(os.path.join(path, 'clusters.json'), 'w') as f:
            json.dump(cluster_labels, f)
        with open(os.path.join(path, 'scores.json'), 'w') as f:
            json.dump(score_list, f)
        
        print('Success!\n')
    print('\n')
        

if __name__ == '__main__':
    
    # Load in command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('qid', type=str, help='the question ID')
    parser.add_argument('--sids_file_path', type=str, 
                        help='file path to a .txt file of student IDs, with each ID on a separate line')
    parser.add_argument('--scorers', type=str, 
                        help='a comma-separated string of scorer names to be used in clustering', 
                        default='AndersonDarlingScorer')
    parser.add_argument('--sizes', type=str, help='a comma-separated string of sizes to be used in clustering',
                       default='25600')
    parser.add_argument('--n_clusters', type=int, help='the number of clusters to create', default=20)
    parser.add_argument('--test_suites_path', type=str, default=None,
                        help='the path to a .json file of test_label: test_args mappings')
    parser.add_argument('--max_parallel', type=int, default=10, 
                        help='maximum number of parallel processes for sampling')
    
    args = parser.parse_args()
    qid = args.qid
    
    # Set model parameters to variables
    config = configparser.ConfigParser()
    if os.path.isfile(os.path.join(DATA_DIR, qid, 'setup', 'config.ini')):   
        config.read(os.path.join(DATA_DIR, qid, 'setup', 'config.ini'))
        
    dtype = config['Parameters']['dtype']
    func_name = config['Parameters']['func_name']
    n_soln_samples = int(config['Parameters']['n_soln_samples'])
    scorers = [scorer.strip() for scorer in args.scorers.split(',')]
    sizes = [int(size.strip()) for size in args.sizes.split(',')]
    if 'proj_method' in config['Parameters']:
        proj_method = config['Parameters']['proj_method']
    else:
        proj_method = ''
    if args.test_suites_path is not None:
        with open(args.test_suites_path) as f:
            test_suites = json.load(f)
    elif os.path.isfile(os.path.join(DATA_DIR, qid, 'setup', 'grading_arguments.json')):
        with open(os.path.join(DATA_DIR, qid, 'setup', 'grading_arguments.json')) as f:
            test_suites = json.load(f)
    else:
        test_suites = {'': []}
    
    # Determining which student IDs to cluster
    print('Gathering student IDs.')
    if args.sids_file_path:
        with open(args.sids_file_path) as f:
             sids = f.readlines()
    else:
        sids = os.listdir(os.path.join(DATA_DIR, qid, 'students'))
    
    # Clustering
    cluster(sids, qid, scorers, sizes, dtype, func_name, n_soln_samples, args.n_clusters,
            max_parallel=args.max_parallel, proj_method=proj_method, test_suites=test_suites)
    