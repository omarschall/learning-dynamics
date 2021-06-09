#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('/scratch/oem214/vanilla-rtrl/')
from cluster import submit_jobs
from continual_learning import *
from core import *
from dynamics import *
from functions import *
from gen_data import *
from learning_algorithms import *
from optimizers import *
from plotting import *
from wrappers import *


### --- Set up all configs --- ###
from itertools import product
n_seeds = 10
macro_configs = config_generator(param_1=[1, 2],
                                 param_2=[10, 100])
micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

### --- Select particular config --- ###
try:
    i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
except KeyError:
    i_job = 0
params, i_seed = micro_configs[i_job]
i_config = i_job//n_seeds

new_random_seed_per_condition = True
if new_random_seed_per_condition:
    np.random.seed(i_job)
else: #Match random seeds across conditions
    np.random.seed(i_seed)


