# Run file for Analysis 8: Aggregate level class variables, block analysis for multicollinearity
import os
import datetime as dt
import numpy as np
import pandas as pd
from Functions.Set_params import set_params_do_initial_preprocess
from Initial_Preprocess import preprocess_drop_DV_T1, dependent_variable

anal = "Analysis_8"
start = dt.datetime.now()

global_seed = np.random.seed(93)
block_file = "Data/MetaData/Variable_blocks/variable_blocks_final.csv"

# ----------------------------
Do_initial_preprocess = False
use_preoptimised_model = True
Test = False
preoptimised_model = 5
Choose_drop_DV_T1 = False
perm_cut_off = 0.01
# ----------------------------

if Choose_drop_DV_T1 == True:
    analysis = "{}_without_{}_T1".format(anal, dependent_variable)
elif Choose_drop_DV_T1 == False:
    analysis = "{}_with_{}_T1".format(anal, dependent_variable)
else:
    print("add True or False as param")
    analysis = None
    breakpoint()

if Test == True:
    t = "_test"
if Test == False:
    t = ""

version = set_params_do_initial_preprocess(Do_initial_preprocess=Do_initial_preprocess,
                                           Choose_drop_DV_T1=Choose_drop_DV_T1,
                                           preprocess_drop_DV_T1=preprocess_drop_DV_T1,
                                           dependent_variable=dependent_variable)

if use_preoptimised_model == True:
    print("Using pre-optimised model {}".format(preoptimised_model))
    if version == "_without_sges_T1":
        start_string = "_26_Aug_2022__14.13"
    elif version == "_with_sges_T1":
        start_string = "_27_Aug_2022__14.22"
    else:
        start_string = None
        breakpoint()
elif use_preoptimised_model == False:
    start_string = start.strftime('_%d_%b_%Y__%H.%M{}'.format(t))
else:
    print("add True or False as value for param use_preoptimised_model")
    start_string = None
    breakpoint()

analysis_path = "{}/".format(analysis)
runtime_path = analysis_path + "Outputs/Runtimes/"
runtime_file = "{}_{}.txt".format(analysis, start_string)
perm_cut_off = 0.01
global_seed = np.random.seed(93)
# -----------------------------------------------------------
if __name__ == "__main__":
    print('Running {} (Aggregate level class variables, block analysis for multicollinearity)'.format(analysis))

    # Preprocess
    if Do_initial_preprocess == True:
        exec(open("Initial_Preprocess.py").read())
        end_preprocess = dt.datetime.now()
        preprocess_runtime = end_preprocess - start
        print("Preprocess runtime: {}".format(preprocess_runtime), file=open(runtime_path + runtime_file, "w"))

    # Prediction
    if use_preoptimised_model == False:
        start_prediction = dt.datetime.now()
        exec(open("Analysis_8a_Prediction.py").read())
        end_prediction = dt.datetime.now()
        prediction_runtime = end_prediction - start_prediction
        print("Prediction runtime: {}".format(prediction_runtime), file=open(runtime_path + runtime_file, "a"))

        # Plot prediction results
        start_plot_pred = dt.datetime.now()
        exec(open("Analysis_8b_Plot_Prediction_Results.py").read())
        end_plot_pred = dt.datetime.now()
        plot_pred_runtime = end_plot_pred - start_plot_pred
        print("Print prediction results runtime: {}".format(plot_pred_runtime),
              file=open(runtime_path + runtime_file, "a"))

    # Permutation Importance
    start_perm = dt.datetime.now()
    exec(open("Analysis_8c_Block_Permutation_Importance.py").read())
    end_perm = dt.datetime.now()
    perm_runtime = end_perm - start_perm
    print("Permutation importance runtime: {}".format(perm_runtime), file=open(runtime_path + runtime_file, "a"))

    # Plot Importance Results
    start_plot_imp = dt.datetime.now()
    exec(open("Analysis_8d_Plot_Importance.py").read())
    end_plot_imp = dt.datetime.now()
    plot_imp_runtime = end_plot_imp - start_plot_imp
    print("Plot importance runtime: {}".format(plot_imp_runtime), file=open(runtime_path + runtime_file, "a"))

    end = dt.datetime.now()
    total = end - start
    print('{} complete. Total time taken: {}'.format(analysis, total), file=open(runtime_path + runtime_file, "a"))
    print('{} complete. Total time taken: {}'.format(analysis, total))