# Run file for predict X years ahead (copy of Analysis 3). Data adjusted for multicollinearity (i.e. base predition model is same as A2)

import datetime as dt
import os

import numpy as np
from Predict_X_years_ahead_with_sges_T1.Parameters.Grid import enet_param_grid, test_enet_param_grid, rf_param_grid, \
        test_rf_param_grid
from Functions.Set_params import set_params_do_initial_preprocess, set_params_testrun
from Initial_Preprocess import preprocess_drop_DV_T1, dependent_variable
from Fixed_params import block_file

start = dt.datetime.now()
anal = "Predict_X_years_ahead"

# ~~~~~~~~~~~~ Set params ~~~~~~~~~~~~
# changeable:
Additional_pred_checks = False
Additional_pred_checks_only = False

Do_initial_preprocess = False
anal_level = "block"
Create_Data = False
Test = False
use_preoptimised_model = True
preoptimised_model = 2
fixed_dummy_r2 = False
run_permutation_importance = False
Choose_drop_DV_T1 = False

# set:
add_teacher_vars = True
if Choose_drop_DV_T1 == True:
    print("****** CHECK: are you sure you want to drop DV Time 1?")
if add_teacher_vars == False:
    print("****** CHECK: are you sure you don't want to add teacher variables?")
# ================================================================================

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
        # start_string = "_29_Nov_2022__09.32{}".format(t)
        # start_string = "_05_Dec_2022__15.22{}".format(t)
        # start_string = "_10_Dec_2022__14.37{}".format(t)
        # start_string = "_12_Dec_2022__09.27{}".format(t)
        # start_string = "_23_Dec_2022__12.04{}".format(t)
        # start_string = "_02_Jan_2023__12.06{}".format(t)
        # start_string = "_14_Jan_2023__21.20{}".format(t)
        # start_string = "_25_Jan_2023__12.32"
        start_string = "_26_Jun_2023__09.13"
elif use_preoptimised_model == False:
    start_string = start.strftime('_%d_%b_%Y__%H.%M{}'.format(t))
else:
    start_string = "error"

enet_params = params["Enet"]
rf_params = params["RF"]

final_data_path = analysis + "/Processed_Data/Processed_Multicollinearity/"
directory_bits = os.fsencode(final_data_path)

analysis_path = "{}/".format(analysis)
global_seed = np.random.seed(93)
runtime_path = analysis_path + "Outputs/Runtimes/"
runtime_file = "{}_{}.txt".format(analysis, start_string)
perm_cut_off = 0.01
# -----------------------------------------------------------
if __name__ == "__main__":
    print('Running {} (Prediction = Individual level variables only, adjustments for multicollinearity)'.format(analysis))

    # Initial Preprocess
    if Do_initial_preprocess == True:
        exec(open("Initial_Preprocess.py").read())
        end_preprocess = dt.datetime.now()
        preprocess_runtime = end_preprocess - start
        print("Preprocess runtime: {}".format(preprocess_runtime), file=open(runtime_path + runtime_file, "w"))

    # Create Data
    if Create_Data == True:
        exec(open("Analysis_Predict_X_years_ahead_1a_Create_Data.py").read())
        end_preprocess = dt.datetime.now()
        preprocess_runtime = end_preprocess - start
        print("Preprocess runtime: {}".format(preprocess_runtime), file=open(runtime_path + runtime_file, "w"))

    # Re-process data for multicollinearity
    if Create_Data == True:
        exec(open("Analysis_Predict_X_years_ahead_1b_Process_Multicolinearity.py").read())
        end_preprocess = dt.datetime.now()
        preprocess_runtime = end_preprocess - start
        print("Preprocess runtime: {}".format(preprocess_runtime), file=open(runtime_path + runtime_file, "w"))

    if use_preoptimised_model == False:
        exec(open("Analysis_Predict_X_years_ahead_2_Prediction.py").read())

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

    # Perm Importance
    if (run_permutation_importance == True) and (anal_level == "block"):
        start_perm = dt.datetime.now()
        exec(open("Analysis_Predict_X_years_ahead_5_Block_Permutation_Importance.py").read())
        end_perm = dt.datetime.now()
        perm_runtime = end_perm - start_perm
        print("Permutation importance runtime: {}".format(perm_runtime), file=open(runtime_path + runtime_file, "a"))

        # Plot Perm Importance Results
        start_plot_imp = dt.datetime.now()
        exec(open("Analysis_Predict_X_years_ahead_6_Plot_Perm_Importance.py").read())
        end_plot_imp = dt.datetime.now()
        plot_imp_runtime = end_plot_imp - start_plot_imp
        print("Plot importance runtime: {}".format(plot_imp_runtime), file=open(runtime_path + runtime_file, "a"))

    #     todo! add plots

    end = dt.datetime.now()
    total = end - start
    print('{} complete. Total time taken: {}'.format(analysis, total), file=open(runtime_path + runtime_file, "a"))
    print('{} complete. Total time taken: {}'.format(analysis, total))
