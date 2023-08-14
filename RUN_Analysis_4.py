# Run file for Analysis 4

import datetime as dt
import numpy as np
from Analysis_4_with_sges_T1.Parameters.Grid import enet_param_grid, test_enet_param_grid, rf_param_grid, test_rf_param_grid
from Functions.Set_params import set_params_do_initial_preprocess, set_params_testrun
from Initial_Preprocess import add_teacher_vars, dependent_variable, preprocess_drop_DV_T1
from Fixed_params import block_file, drop_p_degree_expect, drop_houspan

anal = "Analysis_4"
start = dt.datetime.now()

global_seed = np.random.seed(93)
perm_cut_off = 0.01

Do_initial_preprocess = True
use_preoptimised_model = True
Test = False
preoptimised_model = 2
Choose_drop_DV_T1 = False
Include_T5 = False

# variable block options:

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

t, params = set_params_testrun(test=Test, enet_param_grid= enet_param_grid,
                               test_enet_param_grid=test_enet_param_grid,
                               rf_param_grid=rf_param_grid,
                               test_rf_param_grid=test_rf_param_grid)

if use_preoptimised_model == True:
    if preoptimised_model == 1:
        # start_string = "_29_Jul_2022__09.54"
        # start_string = "_04_Nov_2022__09.05"
        # start_string = "_02_Jan_2023__12.07"
        # start_string = "_14_Jan_2023__21.19"
        start_string = "_25_Jan_2023__12.15"
    if preoptimised_model == 2:
        # start_string = "_26_Aug_2022__14.09"
        # start_string = "_04_Nov_2022__09.05"
        # start_string = "_02_Jan_2023__12.07"
        # start_string = "_14_Jan_2023__21.19"
        start_string = "_25_Jan_2023__12.15"
    print("Using pre-optimised model {}".format(preoptimised_model))
if use_preoptimised_model == False:
    start_string = start.strftime('_%d_%b_%Y__%H.%M{}'.format(t))

enet_params = params["Enet"]
rf_params = params["RF"]

analysis_path = "{}/".format(analysis)
runtime_path = analysis_path + "Outputs/Runtimes/"
runtime_file = "{}_{}.txt".format(analysis, start_string)
# -----------------------------------------------------------
if __name__ == "__main__":
    print('Running {} (Individual level variables only, block analysis for multicollinearity)'.format(analysis))

    # Preprocess
    if Do_initial_preprocess == True:
        exec(open("Initial_Preprocess.py").read())
        end_preprocess = dt.datetime.now()
        preprocess_runtime = end_preprocess - start
        print("Preprocess runtime: {}".format(preprocess_runtime), file=open(runtime_path + runtime_file, "w"))

    # # Prediction
    # start_prediction = dt.datetime.now()
    # exec(open("Analysis_4a_Prediction.py").read())
    # end_prediction = dt.datetime.now()
    # prediction_runtime = end_prediction - start_prediction
    # print("Prediction runtime: {}".format(prediction_runtime), file=open(runtime_path + runtime_file, "a"))
    #
    # # Plot prediction results
    # start_plot_pred = dt.datetime.now()
    # exec(open("Analysis_4b_Plot_Prediction_Results.py").read())
    # end_plot_pred = dt.datetime.now()
    # plot_pred_runtime = end_plot_pred - start_plot_pred
    # print("Print prediction results runtime: {}".format(plot_pred_runtime),
    #       file=open(runtime_path + runtime_file, "a"))

    # Permutation Importance
    start_perm = dt.datetime.now()
    exec(open("Analysis_4c_Block_Permutation_Importance.py").read())
    end_perm = dt.datetime.now()
    perm_runtime = end_perm - start_perm
    print("Permutation importance runtime: {}".format(perm_runtime), file=open(runtime_path + runtime_file, "a"))

    # Plot Importance Results
    start_plot_imp = dt.datetime.now()
    exec(open("Analysis_4d_Plot_Importance.py").read())
    end_plot_imp = dt.datetime.now()
    plot_imp_runtime = end_plot_imp - start_plot_imp
    print("Plot importance runtime: {}".format(plot_imp_runtime), file=open(runtime_path + runtime_file, "a"))

    end = dt.datetime.now()
    total = end - start
    print('{} complete. Total time taken: {}'.format(analysis, total), file=open(runtime_path + runtime_file, "a"))
    print('{} complete. Total time taken: {}'.format(analysis, total))