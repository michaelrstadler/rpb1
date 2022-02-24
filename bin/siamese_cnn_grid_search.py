#!/usr/bin/env python

"""siamese_cnn_grid_search.py: Train a siamese CNN with varying hyperparameters

To run on savio: 
export PYTHONPATH="$PWD/rpb1/bin/cnn_models";module unload python/3.7;module load ml/tensorflow/2.5.0-py37
# Put path to cnn_models here...I don't know how to manage this with savio module system yet (facepalm).
"""

__author__      = "Michael Stadler"
__copyright__   = "Copyright 2022, California, USA"

import subprocess

##################################################################### 
# Set these:

train_siames_cnn_path = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/bin/train_siamese_cnn.py'
data_folder_path = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/results/testsims_uPoNivMJ_10'

# Iterated parameters.
initial_learning_rates = [0.001, 0.0005, 0.0001]
learning_rate_exps = [0.05, 0.1, 0.25]

curriculum_params = (
    # epoch lengths, lower margins, upper margins, name
    ('10', '0', '100', 'nocurriculum'),
    ('10,10,10', '0,0,0', '100,66,33', 'all_to_hard'),
    ('10,10,10', '66,33,0', '100,100,100', 'easy_to_all'),
    ('10,10,10', '66,33,0', '100,66,33', 'easy_to_hard')
)

# Constant parameters
z = '2' # datset size
y = '18' # model layers
c = '2' # learning rate contant for c epochs

##################################################################### 

for r in initial_learning_rates:
    for R in learning_rate_exps:
        r = str(r)
        R = str(R)
        for e,w,u,n in curriculum_params:
            # Note: all args to subprocess.call must be type string.
            name = '_'.join([r, R, z, y, c, n])
            subprocess.call(
                [
                    'python',
                    train_siames_cnn_path,
                    '-f', data_folder_path, # folder
                    '-n', name, # model name
                    '-e', e, # num epochs
                    '-z', z, # dataset size
                    '-y', y, # model layers
                    '-r', r, # initial learning rate
                    '-R', R, # Learning rate exponent
                    '-c', c, # learning rate contant for c epochs
                    '-w', w, # lower margins
                    '-u', u # upper margins
                ]
            )