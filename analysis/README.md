### Code for tha analysis of the data   


       1. analysis/behavior/ : code for behavioral metrics and analysis.
               a. analysis/behavior/measures.py : code for aggregate behavioral measures extracted from the participants
               b. analysis/behavior/behavior_utils: code for utility functions for behavioral metrics
               c. analysis/behavior/datautils: code for utility functions for loading experimental data

       2. analysis/models/ : contains the code for model fitting and analysis.
            
               a. analysis/models/model.py : code for the common model class for all algorithms
               b. analysis/models/momentum.py: code for the TD-momentum model
               c. analysis/models/prospective.py: code for the prospective model
               d. analysis/models/retrospective.py: code for the retrospective model
               e. analysis/models/hybrid.py: code for the hybrid model
               f. analysis/models/td_persistence.py: code for the TD-persistence model
               g. analysis/models/prospective_*: code for running variants of the prospective model
               h. analysis/models/rescorla.py: code for running the Rescorla-Wagner model
               i. analysis/models/run_model.py: code for running model simulations


       3. analysis/generate/ : code for generating configuration files for experiments
            
               a. analysis/generate/json: json files for running experiments
               b. analysis/generate/generate_experiment_trials.py: code for generating blocks and rounds for experiments

