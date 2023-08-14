# Run file for predict X years ahead (copy of Analysis 3). Data adjusted for multicollinearity (i.e. base predition model is same as A2)

import datetime as dt
import os

import numpy as np
from Predict_X_years_ahead_with_sges_T1.Parameters.Grid import enet_param_grid, test_enet_param_grid, rf_param_grid, \
        test_rf_param_grid
from Functions.Set_params import set_params_do_initial_preprocess, set_params_testrun
from Initial_Preprocess import preprocess_drop_DV_T1, dependent_variable

start = dt.datetime.now()
anal = "Predict_X_years_ahead_Analysis_3"

anal_level = "block"
Create_Data = False
Choose_drop_DV_T1 = False
Test = False
use_preoptimised_model = True

if Choose_drop_DV_T1 == True:
    analysis = "{}_without_{}_T1".format(anal, dependent_variable)
elif Choose_drop_DV_T1 == False:
    analysis = "{}_with_{}_T1".format(anal, dependent_variable)
else:
    print("add True or False as param")
    analysis = None
    breakpoint()

t, params = set_params_testrun(test=Test, enet_param_grid= enet_param_grid,
                               test_enet_param_grid=test_enet_param_grid,
                               rf_param_grid=rf_param_grid,
                               test_rf_param_grid=test_rf_param_grid)

version = set_params_do_initial_preprocess(Do_initial_preprocess=Create_Data,
                                           Choose_drop_DV_T1=Choose_drop_DV_T1,
                                           preprocess_drop_DV_T1=preprocess_drop_DV_T1,
                                           dependent_variable=dependent_variable)

if use_preoptimised_model == True:
    print("Using a pre-optimised model")
    if version == "_with_sges_T1":
        start_string = "_29_Nov_2022__09.32{}".format(t)
if use_preoptimised_model == False:
    start_string = start.strftime('_%d_%b_%Y__%H.%M{}'.format(t))

enet_params = params["Enet"]
rf_params = params["RF"]

data_path = "Data/Predict_x_years_ahead/"
directory_bits = os.fsencode(data_path)

analysis_path = "{}/".format(analysis)
global_seed = np.random.seed(93)
runtime_path = analysis_path + "Outputs/Runtimes/"
runtime_file = "{}_{}.txt".format(analysis, start_string)
perm_cut_off = 0.01
# -----------------------------------------------------------
if __name__ == "__main__":
    print('Running {} (Prediction = Individual level variables only, adjustments for multicollinearity)'.format(analysis))

    # Create data
    if Create_Data == True:
        exec(open("Analysis_Predict_X_years_ahead_1_Create_Data_old.py").read())
        end_preprocess = dt.datetime.now()
        preprocess_runtime = end_preprocess - start
        print("Preprocess runtime: {}".format(preprocess_runtime), file=open(runtime_path + runtime_file, "w"))

    if use_preoptimised_model == False:
        exec(open("Analysis_Predict_X_years_ahead_A2b_Prediction.py").read())

    # Plot prediction results
    start_plot_pred = dt.datetime.now()
    exec(open("Analysis_Predict_X_years_ahead_3_Plot_Prediction_Results.py").read())
    end_plot_pred = dt.datetime.now()
    plot_pred_runtime = end_plot_pred - start_plot_pred
    print("Print prediction results runtime: {}".format(plot_pred_runtime), file=open(runtime_path + runtime_file, "a"))

    # SHAP Importance
    start_perm = dt.datetime.now()
    exec(open("Analysis_Predict_X_years_ahead_4_SHAP_Importance.py").read())
    end_perm = dt.datetime.now()
    perm_runtime = end_perm - start_perm
    print("Permutation importance runtime: {}".format(perm_runtime), file=open(runtime_path + runtime_file, "a"))

    end = dt.datetime.now()
    total = end - start
    print('{} complete. Total time taken: {}'.format(analysis, total), file=open(runtime_path + runtime_file, "a"))
    print('{} complete. Total time taken: {}'.format(analysis, total))
