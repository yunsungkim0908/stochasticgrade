import json
import numpy as np
import os
import sklearn.cluster
import sklearn.metrics
import sklearn.mixture

from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from stochasticgrade.constants import *


def get_euclidean_distances(stud_samples, soln_samples, sid, qid, test_label='', dilation=1, anchor_id=None):
    """
    Measures the Euclidean distance between a randomly selected 
    anchor point (near the solution distribution) and the student samples.
    """

    if anchor_id is None:
        anchor_path = os.path.join(DATA_DIR, qid, 'setup', 'random_anchor.json')
    else:
        anchor_path = os.path.join(DATA_DIR, qid, 'setup', f'random_anchor{anchor_id}.json')
    
    # Sample the random anchor point from a bounding box of the solution samples
    if not os.path.isfile(anchor_path):
        dirname = '/'.join(anchor_path.split('/')[:-1])
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        minima = np.amin(soln_samples, axis=0)
        maxima = np.amax(soln_samples, axis=0)
        midpoints = (minima + maxima) / 2
        adjusted_minima = midpoints - (midpoints - minima) * (1 + dilation)
        adjusted_maxima = midpoints + (maxima - midpoints) * (1 + dilation)
        
        sample_anchor = True
        while sample_anchor:
            anchor = [np.random.uniform(x[0], x[1]) for x in zip(adjusted_minima, adjusted_maxima)]
            for i in range(len(anchor)):
                if anchor[i] < minima[i] or anchor[i] > maxima[i]:
                    sample_anchor = False
                    
        with open(os.path.join(anchor_path), 'w') as f:
            json.dump(anchor, f)
        soln_dists = np.sqrt(np.sum((soln_samples - anchor) * (soln_samples - anchor), axis=1)) 

        if anchor_id is None:
            path = os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 'euclidean_distances.npy')
        else:
            path = os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 
                                f'euclidean_distances{anchor_id}.npy')
        np.save(path, soln_dists)
        
    with open(os.path.join(anchor_path)) as f:
        anchor = json.load(f)
            
    # Compute the distances of student samples points to the anchor point
    dists = np.sqrt(np.sum((stud_samples - anchor) * (stud_samples - anchor), axis=1))
    
    if 'solution' in sid:
        sid_type = 'solution/mc_solutions' if 'mc_solution' in sid else 'solution'
    else:
        sid_type = 'students'

    if anchor_id is None:
        path = os.path.join(DATA_DIR, qid, sid_type, sid, 'euclidean_distances.npy')
    else:
        path = os.path.join(DATA_DIR, qid, sid_type, sid, f'euclidean_distances{anchor_id}.npy')
    np.save(path, dists)
    
    return dists


def get_orthogonal_projections(stud_samples, soln_samples, sid, qid, test_label='', vector_id=None):
    """
    Calculates the projection of the student samples onto a unit vector
    from the corresponding n-dimensional hypersphere. 
    """

    if vector_id is None:
        vector_path = os.path.join(DATA_DIR, qid, 'setup', 'random_unit_vector.json')
    else:
        vector_path = os.path.join(DATA_DIR, qid, 'setup', f'random_unit_vector{vector_id}.json')
    
    # Sample the random unit vector from the hypersphere
    if not os.path.isfile(vector_path):
        dirname = '/'.join(vector_path.split('/')[:-1])
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            
        n = soln_samples.shape[1:]
        point = np.random.normal(size=n)
        norm = np.linalg.norm(point)
        unit_vector = point / norm

        with open(os.path.join(vector_path), 'w') as f:
            json.dump(unit_vector.tolist(), f)
        soln_dists = np.dot(soln_samples, unit_vector)

        if vector_id is None:
            path = os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 'orthogonal_projections.npy')
        else:
            path = os.path.join(DATA_DIR, qid, 'solution', 'solution', test_label, 
                                f'orthogonal_projections.npy{vector_id}')
        np.save(path, soln_dists)
        
    with open(os.path.join(vector_path)) as f:
        unit_vector = json.load(f)
        
    # Compute the distances of student samples points to the anchor point
    dists = np.dot(stud_samples, unit_vector)
    
    if 'solution' in sid:
        sid_type = 'solution/mc_solutions' if 'mc_solution' in sid else 'solution'
    else:
        sid_type = 'students'

    if vector_id is None:
        path = os.path.join(DATA_DIR, qid, sid_type, sid, 'orthogonal_projections.npy')
    else:
        path = os.path.join(DATA_DIR, qid, sid_type, sid, f'orthogonal_projections{vector_id}.npy')
    np.save(path, dists)
    
    return dists