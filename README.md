# StochasticGrade

This repository contains the code for StochasticGrade, an automated assessment framework for evaluating stochastic programs. StochasticGrade is introduced in the paper "Grading and Clustering Student Programs That Produce Probabilistic Output," published in EDM '24: the 17th International Conference on Educational Data Mining, 2024.


## Installation

1. Clone this repository and navigate to the cloned directory:
    ```sh
    git clone [repository-url]
    cd [repository-directory]
    ```
2. Install the package in editable mode:
    ```sh
    pip install -e .
    ```
3. Adjust the `*_DIR` values in `stochasticgrade/constants.py` according to your environment.

## Organizing Data for Your Problem

Create a folder containing the following information:

- `solution.py`: The stochastic program that provides the correct implementation of the problem.

- Student data, formatted as either:
  - `[student-id].py` files: Each file contains a stochastic program written by a student. Each student must have a unique `student-id`.
  - A `.json` file: Contains a mapping between each `student-id` and their stochastic program as a string.

- (Optional) `grading_arguments.json`: Used if the program to be evaluated requires input arguments. This file is a mapping between a unique label and the input arguments. Refer to `data/TS/setup/grading_arguments.json` for an example.

- (Optional) `closest_error.py`: The buggy student program most similar to the correct answer. This is used by `choose_n.py` to determine the minimum sample size that adheres to a provided false acceptance rate.


## Creating the Problem

1. After organizing your problem data, set up the problem by configuring the parameters of StochasticGrade through the `preprocess.py` script. From the `stochasticgrade/stochasticgrade` directory, run:
    ```sh
    python preprocess.py [question-id] [data-path] [function-name] [additional-flags]
    ```
    - `qid`: A unique identifier for the problem you are creating.
    - `data-path`: The path to the directory containing your organized problem data.
    - `function-name`: The name of the function to be sampled from the solution and student programs.
    - `additional-flags`: Any additional flags needed for your specific problem setup (described below).
  
    This creates the `data` directory if it does not exist, which contains the `qid` directory and its `students`, `solution`, `setup`, and `results` subdirectories.

    If necessary, adjust the default StochasticGrade parameters to fit the nature of your problem. Available flags include:

    - `--scorer`: Scoring method. Default is `AndersonDarlingScorer`.
    - `--min_n`: Minimum number of samples. Default is 400.
    - `--max_n`: Maximum number of samples. Default is 409600.
    - `--n_scale_factor`: Scale factor for the number of samples. Default is 4.
    - `--frr`: The false rejection rate for the problem. Default is 0.01.

    Additional parameters may need to be adjusted:
    
    - `--dtype`: The data type of the output from the solution problem. This is either `scalar`, `list`, or `array_shape_[SHAPE_TUPLE]` for an array of shape `SHAPE_TUPLE`. Default is `scalar`.
    - `--n_soln_samples`: Number of solution samples. Default is 500000.
    - `--proj_method`: Projection method used for multidimensional samples. Either `ED` for Euclidean Distance or `OP` for Orthogonal Projection. Default is `ED`. 
    - `--M`: Number of iterations for Monte Carlo sampling. Default is 1000.
    - `--max_parallel`: Maximum number of parallel processes for sampling. Default is 20.
    - `--save_mc_samples`: Flag to save Monte Carlo samples.

3. (Optional) If you would like to guarantee a provided false acceptance rate, ensure that you have uploaded `closest_error.py` and that you are using the AndersonDarlingScorer or WassersteinScorer. Then, run
    ```sh
    python choose_n.py [qid] [additional-flags]
    ```
    Additional flags can be set as:
    - `--far`: The false acceptance rate (FAR). Default is 0.1.
    - `--M`: The number of times to execute the grading algorithm to check the FAR.
    - `--max_parallel`: The maximum number of parallel processes for sampling.


## Grading the Problem 

Grade the student submissions to the problem by running

  ```sh
  python stochastic_grade.py [qid] [additional-flags]
  ```
  
  Set additional flags as:
  - `--sids_file_path`: If you don't want to grade all students, provide a file path to a .txt file of student IDs to grade, with each ID on a separate line.
  - `--max_parallel`: Maximum number of parallel processes for grading.
  - `--delete_samples`: Delete samples/projections after grading for space conservation
  - `--no_best_n`: Use the max value of N from `config.ini` instead of `best_n` (if `choose_n.py` was run).

Results are stored under the `results` directory for the `qid`. The individual result for each student can be found in the `students` directory.

## Clustering Student Responses

If you'd like to cluster student programs, run
  ```sh
  python cluster.py [qid] [additional-flags]
  ```

  Set cluster-specific flags as:
  - `--scorers`: A comma-separated string of scorer names to be used in clustering (e.g. AndersonDarlingScorer,TScorer). Default is AndersonDarlingScorer. 
  - `--sizes`: A comma-separated string of sizes to be used in clustering (e.g. 400,1600,6400). Default is 25600.
  - `--n_clusters`: The number of clusters to create. This works best if it coincides roughly with the expected number of errors. Default is 20.
  - `--normalize_scores`: Normalize the calculated scores before clustering.
  - `--n_projections`: If you would like to cluster with mulitple projections, the number of new projections to generate for multidimensional samples. Default is 0.
  - `--test_suites_path`: If you would like to cluster with multiple input cases, the path to a .json file mapping of labels to input arguments.
  - `--max_parallel`: Same as above.
  - `--sids_file_path`: Same as above.

Results are stored under the `results/clusters` directory for the `qid`.


## Examples

You can find four sample problem directories in the repository. Here are the grading pipelines for all four:

### SimExp

```sh
python preprocess.py SimExp ./example_data/SimExp/ simulate_exponential
python stochastic_grade.py SimExp
```

### AlgoArt 

```sh
python preprocess.py AlgoArt ./example_data/AlgoArt/ main --dtype=list --scorer=MSDScorer
python stochastic_grade.py AlgoArt
```

### TS
```sh
python preprocess.py TS ./example_data/TS/ main --scorer=TScorer
python stochastic_grade.py TS
```

### RN
```sh
python preprocess.py RN ./example_data/RN/ main --dtype=array_shape_(10,) --scorer=TScorer --proj_method=ED
python stochastic_grade.py RN
```

## Troubleshooting

- A common source of error is incorrectly specifying model parameters for StochasticGrade. Make sure to fully specify any necessary parameters for the model.
- You may encounter a problem using the `numpy.random` module if using parallelized grading, where student samples are generated using a similar seed. It is preferred to use the `random` module. Alternatively, set the seed based on the process ID, such as `os.getpid() % 2**32`. 
