# Run file for predict X years ahead (copy of Analysis 3). Data adjusted for multicollinearity (i.e. base predition model is same as A2)

import datetime as dt
import os

import numpy as np
from Predict_same_year.Parameters.Grid import enet_param_grid, test_enet_param_grid, rf_param_grid, \
        test_rf_param_grid
from Functions.Set_params import set_params_do_initial_preprocess, set_params_testrun
from Initial_Preprocess import preprocess_drop_DV_T1, dependent_variable
from Fixed_params import Compare_DV_T1, Choose_drop_DV_T1

Choose_drop_DV_T1 = False
dependent_variable = "sges_T1"

start = dt.datetime.now()
analysis = "Predict_same_year"

Do_initial_preprocess = True
anal_level = "block"
Create_Data = True
Test = False
use_preoptimised_model = True

# fixed
times = [1, 2, 3, 4, 5]
xticks = ["5 --> 5", "6 --> 6", "7 --> 7", "8 --> 8", "9 --> 9"]

t, params = set_params_testrun(test=Test, enet_param_grid= enet_param_grid,
                               test_enet_param_grid=test_enet_param_grid,
                               rf_param_grid=rf_param_grid,
                               test_rf_param_grid=test_rf_param_grid)

if use_preoptimised_model == True:
    print("Using a pre-optimised model")
    # start_string = "_08_Dec_2022__16.51{}".format(t)
    start_string = "_18_Jan_2023__15.51{}".format(t)
elif use_preoptimised_model == False:
    start_string = start.strftime('_%d_%b_%Y__%H.%M{}'.format(t))
else:
    start_string = "error"

enet_params = params["Enet"]
rf_params = params["RF"]

original_data_load_path = "Analysis_2_with_sges_T1/Processed_Data/"
data_save_path = "Predict_same_year/Processed_Data/"
directory_bits = os.fsencode(data_save_path)

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

    # Re-process data for multicollinearity
    if (Do_initial_preprocess == True) and (Create_Data==True):
        exec(open("Analysis_2a_Preprocess_Multicolin.py").read())
        end_preprocess = dt.datetime.now()
        preprocess_runtime = end_preprocess - start
        print("Preprocess runtime: {}".format(preprocess_runtime), file=open(runtime_path + runtime_file, "w"))

    # Create data
    if Create_Data == True:
        # uses preprocessed data from A2 (preprocessed for multicollinearity)
        exec(open("Analysis_Predict_same_year_1_Create_Data.py").read())
        end_preprocess = dt.datetime.now()
        preprocess_runtime = end_preprocess - start
        print("Preprocess runtime: {}".format(preprocess_runtime), file=open(runtime_path + runtime_file, "w"))

    if use_preoptimised_model == False:
        exec(open("Analysis_Predict_same_year_2_Prediction.py").read())

    # Plot prediction results
    start_plot_pred = dt.datetime.now()
    exec(open("Analysis_Predict_same_year_3_Plot_Prediction_Results.py").read())
    end_plot_pred = dt.datetime.now()
    plot_pred_runtime = end_plot_pred - start_plot_pred
    print("Print prediction results runtime: {}".format(plot_pred_runtime), file=open(runtime_path + runtime_file, "a"))

    # SHAP Importance
    start_perm = dt.datetime.now()
    exec(open("Analysis_Predict_same_year_4_SHAP_Importance.py").read())
    end_perm = dt.datetime.now()
    perm_runtime = end_perm - start_perm
    print("Permutation importance runtime: {}".format(perm_runtime), file=open(runtime_path + runtime_file, "a"))

    end = dt.datetime.now()
    total = end - start
    print('{} complete. Total time taken: {}'.format(analysis, total), file=open(runtime_path + runtime_file, "a"))
    print('{} complete. Total time taken: {}'.format(analysis, total))
