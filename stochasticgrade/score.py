"""
score.py
--------
This file stores the disparity functions used for grading.
The pre-defined functions are T, MSD, AD, and WS.
"""

import inspect
import json
import numpy as np
import scipy
import scipy.stats as stats

from abc import ABC, abstractmethod

from stochasticgrade.constants import *


class Scorer(ABC):
    """
    Scores are only based on the samples being evaluated.
    They are independent of scores assigned to other samples.

    Inheriting classes should implement the following functions:
        - __str__
        - compute_score
    """
    def __init__(self):
        pass
        
    @abstractmethod
    def __str__(self):
        """
        Returns:
            Name of the class string (to be used for out_dirname)
        """
        raise NotImplementedError

    @abstractmethod
    def compute_score(self, samples):
        """
        Returns:
            score: the calculated score (double)
        """
        raise NotImplementedError

    
class AndersonDarlingScorer(Scorer):
    """
    Anderson-Darling statistic based scoring method.
    Calculates the Anderson-Darling test statistic between the
    student and solution samples.
    """
    def __init__(self):
        super().__init__()
        self.monte_carlo = False

    def __str__(self):
        return 'AndersonDarlingScorer'

    def compute_score(self, stud_samples, soln_samples):
        np.random.shuffle(soln_samples)
        np.random.shuffle(stud_samples)
        score = stats.anderson_ksamp([soln_samples, stud_samples]).statistic
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        frr_string = np.format_float_positional(frr, trim='-')
        table = {
            '0.000046875': 8.726,
            '0.00009375': 8.150,
            '0.00015625': 7.695,
            '0.0001875': 7.528,
            '0.0003125': 7.054,
            '0.000375': 6.884,
            '0.000625': 6.405,
            '0.00075': 6.235,
            '0.00078125': 6.197,
            '0.00125': 5.760,
            '0.0015': 5.592,
            '0.0015625': 5.555,
            '0.0025': 5.124,
            '0.003': 4.958,
            '0.003125': 4.921,
            '0.005': 4.497,
            '0.00625': 4.297,
            '0.01': 3.878,
            '0.0125': 3.681,
            '0.025': 3.077,
            '0.05': 2.492,
            '0.1': 1.933
        }
        
        if frr_string in table:
            threshold = table[frr_string]
        else:
            print('The provided FRR is not accepted.')
            print('For the AndersonDarlingScorer, you must select 0.003, 0.01, 0.05, or 0.1.')
            raise Exception
        return threshold
    
    
class MSDScorer(Scorer):
    """
    MSD-based scoring method.
    Measures the difference in spread between the student and
    solution samples.
    """
    def __init__(self):
        super().__init__()
        self.soln_var_term = None
        self.monte_carlo = True

    def __str__(self):
        return 'MSDScorer'

    def compute_score(self, stud_samples, soln_samples):
        mean_term = np.linalg.norm(np.mean(stud_samples, axis=0) - np.mean(soln_samples, axis=0)) ** 2
        stud_var_term = np.std(stud_samples) ** 2
        if self.soln_var_term is None:
            self.soln_var_term = np.std(soln_samples) ** 2
        score = abs(mean_term + stud_var_term - self.soln_var_term)
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        with open(monte_carlo_path) as f:
            scores = json.load(f)
        scores = scores[str(num_stud_samples)]
        scores.sort()
        scores.reverse()
        index = int(len(scores) * frr)
        threshold = scores[index]
        return threshold  
    
    
class TScorer(Scorer):
    """
    T-test based scoring method.
    Measures the difference between the student and solution sample means
    by the t-test.
    """
    def __init__(self):
        super().__init__()
        self.monte_carlo = False

    def __str__(self):
        return 'TScorer'

    def compute_score(self, stud_samples, soln_samples):
        ttest = stats.ttest_ind(stud_samples, soln_samples)
        score = ttest.statistic
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        df = num_stud_samples + num_soln_samples - 2
        t_stat = scipy.special.stdtrit(df, 1 - frr)
        threshold = {'lower_threshold': -t_stat, 'upper_threshold': t_stat}
        return threshold   
    

class WassersteinScorer(Scorer):
    """
    Wasserstein distance based scoring method.
    Calculates the Wasserstein distance between the student and 
    solution samples.
    """
    def __init__(self):
        super().__init__()
        self.monte_carlo = True
        
    def __str__(self):
        return 'WassersteinScorer'
    
    def compute_score(self, stud_samples, soln_samples):
        score = stats.wasserstein_distance(stud_samples, soln_samples)
        if not np.isfinite(score):
            score = 1e7
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        with open(monte_carlo_path) as f:
            scores = json.load(f)
        scores = scores[str(num_stud_samples)]
        scores.sort()
        scores.reverse()
        index = int(len(scores) * frr)
        threshold = scores[index]
        return threshold 


def make_scorer_map():
    """ 
    Create a mapping between scorer names and scorers.
    """
    import __main__
    __name__ = "__main__"
    scorer_map = {}

    # Get all classes defined in this module
    classes = inspect.getmembers(__import__(__name__), inspect.isclass)

    for name, cls in classes:
        if issubclass(cls, Scorer) and cls is not Scorer:
            scorer_map[cls.__name__] = cls()

    return scorer_map