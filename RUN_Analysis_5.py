# Run file for Analysis 5
# Aggregate clas features and mean centered individual vars, no adjustments for multicollinearity

import datetime as dt
import numpy as np
import pandas as pd

from Functions.Set_params import set_params_do_initial_preprocess, set_params_testrun
from Initial_Preprocess import preprocess_drop_DV_T1, dependent_variable

start = dt.datetime.now()
# Set params:
Do_initial_preprocess = False
Choose_drop_DV_T1 = False
Test = False
use_preoptimised_model = False

# ================================================================================
anal = "Analysis_5"

if Choose_drop_DV_T1 == True:
    analysis = "{}_without_sges_T1".format(anal)
    from Analysis_6_without_sges_T1.Parameters.Grid import enet_param_grid, test_enet_param_grid, rf_param_grid, \
        test_rf_param_grid
elif Choose_drop_DV_T1 == False:
    analysis = "{}_with_sges_T1".format(anal)
    from Analysis_6_with_sges_T1.Parameters.Grid import enet_param_grid, test_enet_param_grid, rf_param_grid, \
        test_rf_param_grid
else:
    print("add True or False as param")
    analysis = None
    breakpoint()

version = set_params_do_initial_preprocess(Do_initial_preprocess=Do_initial_preprocess,
                                           Choose_drop_DV_T1=Choose_drop_DV_T1,
                                           preprocess_drop_DV_T1=preprocess_drop_DV_T1,
                                           dependent_variable=dependent_variable)

t, params = set_params_testrun(test=Test, enet_param_grid= enet_param_grid,
                               test_enet_param_grid=test_enet_param_grid,
                               rf_param_grid=rf_param_grid,
                               test_rf_param_grid=test_rf_param_grid)

if use_preoptimised_model == True:
    start_string = "_10_Aug_2022__17.13"
    print("Using a pre-optimised model")
if use_preoptimised_model == False:
    start_string = start.strftime('_%d_%b_%Y__%H.%M{}'.format(t))

enet_params = params["Enet"]
rf_params = params["RF"]

analysis_path = "{}/".format(analysis)
data_path = "Data/Initial_Preprocess/Data_with_Aggregate_Features/"
start = dt.datetime.now()
global_seed = np.random.seed(93)
runtime_path = analysis_path + "Outputs/Runtimes/"
runtime_file = "{}_{}.txt".format(analysis, start_string)
perm_cut_off = 0.01
# -----------------------------------------------------------
if __name__ == "__main__":
    print('Running {} (Aggregate class features added, no adjustments for multicollinearity)'.format(analysis))

    # Preprocess
    if Do_initial_preprocess == True:
        exec(open("Initial_Preprocess.py").read())
        end_preprocess = dt.datetime.now()
        preprocess_runtime = end_preprocess - start
        print("Preprocess runtime: {}".format(preprocess_runtime), file=open(runtime_path + runtime_file, "w"))

    # Create aggregate and mean-centered variables
    exec(open("Analysis_5_Preprocess_Aggregate_Variables.py").read())
    end_preprocess = dt.datetime.now()
    preprocess_runtime = end_preprocess - start
    print("Preprocess aggregate variables runtime: {}".format(preprocess_runtime), file=open(runtime_path + runtime_file, "a"))

    # Prediction
    start_prediction = dt.datetime.now()
    exec(open("Analysis_5a_Prediction.py").read())
    end_prediction = dt.datetime.now()
    prediction_runtime = end_prediction - start_prediction
    print("Prediction runtime: {}".format(prediction_runtime), file=open(runtime_path + runtime_file, "a"))

    # Plot prediction results
    start_plot_pred = dt.datetime.now()
    exec(open("Analysis_5b_Plot_Prediction_Results.py").read())
    end_plot_pred = dt.datetime.now()
    plot_pred_runtime = end_plot_pred - start_plot_pred
    print("Print prediction results runtime: {}".format(plot_pred_runtime), file=open(runtime_path + runtime_file, "a"))

    # # Permutation importance
    # start_perm = dt.datetime.now()
    # exec(open("Analysis_5c_Permutation_Importance.py").read())
    # end_perm = dt.datetime.now()
    # perm_runtime = end_perm - start_perm
    # print("Permutation importance runtime: {}".format(perm_runtime), file=open(runtime_path + runtime_file, "a"))
    #
    # # Plot importance results
    # start_plot_imp = dt.datetime.now()
    # exec(open("Analysis_5d_Plot_Importance.py").read())
    # end_plot_imp = dt.datetime.now()
    # plot_imp_runtime = end_plot_imp - start_plot_imp
    # print("Plot importance runtime: {}".format(plot_imp_runtime), file=open(runtime_path + runtime_file, "a"))

    end = dt.datetime.now()
    total = end - start
    print('{} complete. Total time taken: {}'.format(analysis, total), file=open(runtime_path + runtime_file, "a"))
    print('{} complete. Total time taken: {}'.format(analysis, total))