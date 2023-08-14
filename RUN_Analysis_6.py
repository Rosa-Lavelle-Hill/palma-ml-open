# Run file for Analysis 6
# Aggregate and mean centered variables, adjustments for multicollinearity above threshold
import os
import datetime as dt
import numpy as np
from Initial_Preprocess import preprocess_drop_DV_T1, dependent_variable
from Functions.Set_params import set_params_do_initial_preprocess, set_params_testrun

start = dt.datetime.now()
# Set params:
Do_initial_preprocess = False
Choose_drop_DV_T1 = True
Test = True
use_preoptimised_model = False

# ================================================================================
anal = "Analysis_6"

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
    print("Using a pre-optimised model")
    if version == "_without_sges_T1":
        start_string = "_22_Aug_2022__09.16"
    if version == "_with_sges_T1":
        start_string = "_18_Aug_2022__10.03"
if use_preoptimised_model == False:
    start_string = start.strftime('_%d_%b_%Y__%H.%M{}'.format(t))

enet_params = params["Enet"]
rf_params = params["RF"]

analysis_path = "{}/".format(analysis)
global_seed = np.random.seed(93)
runtime_path = analysis_path + "Outputs/Runtimes/"
runtime_file = "{}_{}.txt".format(analysis, start_string)
perm_cut_off = 0.01
multicolinear_threshold = 0.7
# -----------------------------------------------------------
if __name__ == "__main__":
    print('Running {} (Aggreagte class level variables, adjustments for multicollinearity above threshold ({}))'
          .format(analysis, multicolinear_threshold))

    # Initial Preprocess
    if Do_initial_preprocess == True:
        exec(open("Initial_Preprocess.py").read())
        end_preprocess = dt.datetime.now()
        preprocess_runtime = end_preprocess - start
        print("Preprocess runtime: {}".format(preprocess_runtime),
              file=open(runtime_path + runtime_file, "w"))

    # Multicollinearity Preprocess
    exec(open("Analysis_6a_Preprocess_Multicolin.py").read())
    end_preprocess = dt.datetime.now()
    preprocess_runtime = end_preprocess - start
    print("Preprocess runtime: {}".format(preprocess_runtime),
          file=open(runtime_path + runtime_file, "w"))

    # Create aggregate and mean-centered variables
    from Analysis_5_Preprocess_Aggregate_Variables import aggregate_meta_data
    from Functions.Aggregate_Features import Class_level

    data_path = "{}/Processed_Data/".format(analysis)
    directory_bits = os.fsencode(data_path)
    save_path = "{}/Processed_Data/Processed_Aggregate_Features/".format(analysis)

    Class_level(directory_bits=directory_bits,
                data_path=data_path,
                aggregate_meta_data=aggregate_meta_data,
                save_path=save_path,
                version=version,
                save_name="aggregate_preprocessed")

    end_preprocess = dt.datetime.now()
    preprocess_runtime = end_preprocess - start
    print("Preprocess aggregate variables runtime: {}".format(preprocess_runtime),
          file=open(runtime_path + runtime_file, "a"))

    # Prediction
    if use_preoptimised_model == False:
        start_prediction = dt.datetime.now()
        exec(open("Analysis_6b_Prediction.py").read())
        end_prediction = dt.datetime.now()
        prediction_runtime = end_prediction - start_prediction
        print("Prediction runtime: {}".format(prediction_runtime), file=open(runtime_path + runtime_file, "a"))

    # Post-hoc tests
    start_post_hoc_tests = dt.datetime.now()
    exec(open("Analysis_6f_posthoc_tests.py").read())
    end_post_hoc_tests = dt.datetime.now()
    prediction_runtime = end_post_hoc_tests - start_post_hoc_tests
    print("Posthoc tests runtime: {}".format(prediction_runtime), file=open(runtime_path + runtime_file, "a"))

    # Plot prediction results
    start_plot_pred = dt.datetime.now()
    exec(open("Analysis_6c_Plot_Prediction_Results.py").read())
    end_plot_pred = dt.datetime.now()
    plot_pred_runtime = end_plot_pred - start_plot_pred
    print("Print prediction results runtime: {}".format(plot_pred_runtime), file=open(runtime_path + runtime_file, "a"))

    # Permutation Importance
    start_perm = dt.datetime.now()
    exec(open("Analysis_6d_Permutation_Importance.py").read())
    end_perm = dt.datetime.now()
    perm_runtime = end_perm - start_perm
    print("Permutation importance runtime: {}".format(perm_runtime), file=open(runtime_path + runtime_file, "a"))

    # Plot Importance Results
    start_plot_imp = dt.datetime.now()
    exec(open("Analysis_6e_Plot_Importance.py").read())
    end_plot_imp = dt.datetime.now()
    plot_imp_runtime = end_plot_imp - start_plot_imp
    print("Plot importance runtime: {}".format(plot_imp_runtime), file=open(runtime_path + runtime_file, "a"))

    end = dt.datetime.now()
    total = end - start
    print('{} complete. Time taken: {}'.format(analysis, total), file=open(runtime_path + runtime_file, "a"))
    print('{} complete. Time taken: {}'.format(analysis, total))