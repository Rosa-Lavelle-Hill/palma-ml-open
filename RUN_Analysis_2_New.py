# Run file for Analysis 2
# Individual level variables only, adjustments for multicollinearity above threshold

import datetime as dt
import os
from sklearn.metrics import r2_score
import numpy as np
from Initial_Preprocess import preprocess_drop_DV_T1, dependent_variable
from Functions.Set_params import set_params_do_initial_preprocess, set_params_testrun
from Fixed_params import add_teacher_vars

# ~~~~~~~~~~~~ Set params ~~~~~~~~~~~~

Do_initial_preprocess = False
#^if true: then check params in Initial_Preprocess.py
Test = False
use_preoptimised_model = True
fixed_dummy_r2 = False

# variable block options (only one can be true):
change_emotions = False
change_cog = True

if change_cog == change_emotions:
    print("Error: can either use change emotions or change cog, not both")
    breakpoint()

# set:
Choose_drop_DV_T1 = False

if Choose_drop_DV_T1 == True:
    print("****** CHECK: are you sure you want to drop DV Time 1?")
if add_teacher_vars == False:
    print("****** CHECK: are you sure you don't want to add teacher variables?")
# ================================================================================
start = dt.datetime.now()
anal = "Analysis_2"

if Choose_drop_DV_T1 == True:
    analysis = "{}_without_sges_T1".format(anal)
    from Analysis_2_without_sges_T1.Parameters.Grid import enet_param_grid, test_enet_param_grid, rf_param_grid, \
        test_rf_param_grid
elif Choose_drop_DV_T1 == False:
    analysis = "{}_with_sges_T1".format(anal)
    from Analysis_2_with_sges_T1.Parameters.Grid import enet_param_grid, test_enet_param_grid, rf_param_grid, \
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
    if Choose_drop_DV_T1 == False:
        start_string = "_25_Jan_2023__12.15"
        # start_string = "_14_Jan_2023__21.19"
        # start_string = "_02_Jan_2023__12.07"
        # start_string = "_30_Nov_2022__17.22"
        # start_string = "_04_Nov_2022__09.05"
        # start_string = "_27_Aug_2022__14.22"
        # start_string = "_11_Aug_2022__16.01"
    print("Using a pre-optimised model")
if use_preoptimised_model == False:
    start_string = start.strftime('_%d_%b_%Y__%H.%M{}'.format(t))

# todo: put different start strings in for preoptimised model depending on which version running

enet_params = params["Enet"]
rf_params = params["RF"]

final_data_path = analysis + "/Processed_Data/"
directory_bits = os.fsencode(final_data_path)

analysis_path = "{}/".format(analysis)
global_seed = np.random.seed(93)
runtime_path = analysis_path + "Outputs/Runtimes/"
runtime_file = "{}_{}.txt".format(analysis, start_string)
perm_cut_off = 0.01
multicolinear_threshold = 0.7
# -----------------------------------------------------------
if __name__ == "__main__":
    print('Running {} (Individual level variables only, adjustments for multicollinearity above threshold ({}))'
          .format(analysis, multicolinear_threshold))

    # Initial Preprocess
    if Do_initial_preprocess == True:
        exec(open("Initial_Preprocess.py").read())
        end_preprocess = dt.datetime.now()
        preprocess_runtime = end_preprocess - start
        print("Preprocess runtime: {}".format(preprocess_runtime), file=open(runtime_path + runtime_file, "w"))

    # Multicollinearity Preprocess
    exec(open("Analysis_2a_Preprocess_Multicolin.py").read())
    end_preprocess = dt.datetime.now()
    preprocess_runtime = end_preprocess - start
    print("Preprocess runtime: {}".format(preprocess_runtime), file=open(runtime_path + runtime_file, "w"))

    # Prediction
    if use_preoptimised_model == False:
        start_prediction = dt.datetime.now()
        exec(open("Analysis_2b_Prediction_New.py").read())
        end_prediction = dt.datetime.now()
        prediction_runtime = end_prediction - start_prediction
        print("Prediction runtime: {}".format(prediction_runtime), file=open(runtime_path + runtime_file, "a"))

    # if Choose_drop_DV_T1 == False:
    #     # Post-hoc tests
    #     start_post_hoc_tests = dt.datetime.now()
    #     exec(open("Analysis_2f_posthoc_tests.py").read())
    #     end_post_hoc_tests = dt.datetime.now()
    #     prediction_runtime = end_post_hoc_tests - start_post_hoc_tests
    #     print("Posthoc tests runtime: {}".format(prediction_runtime), file=open(runtime_path + runtime_file, "a"))
    # #     ^ only running additional tests to compare sges_T1 when sges is controlled for

    # Plot prediction results
    start_plot_pred = dt.datetime.now()
    exec(open("Analysis_2c_Plot_Prediction_Results.py").read())
    end_plot_pred = dt.datetime.now()
    plot_pred_runtime = end_plot_pred - start_plot_pred
    print("Print prediction results runtime: {}".format(plot_pred_runtime), file=open(runtime_path + runtime_file, "a"))

    # # Permutation Importance
    # start_perm = dt.datetime.now()
    # exec(open("Analysis_2d_Permutation_Importance.py").read())
    # end_perm = dt.datetime.now()
    # perm_runtime = end_perm - start_perm
    # print("Permutation importance runtime: {}".format(perm_runtime), file=open(runtime_path + runtime_file, "a"))
    #
    # # Plot Importance Results
    # start_plot_imp = dt.datetime.now()
    # exec(open("Analysis_2e_Plot_Importance.py").read())
    # end_plot_imp = dt.datetime.now()
    # plot_imp_runtime = end_plot_imp - start_plot_imp
    # print("Plot importance runtime: {}".format(plot_imp_runtime), file=open(runtime_path + runtime_file, "a"))

    end = dt.datetime.now()
    total = end - start
    print('{} complete. Time taken: {}'.format(analysis, total), file=open(runtime_path + runtime_file, "a"))
    print('{} complete. Time taken: {}'.format(analysis, total))