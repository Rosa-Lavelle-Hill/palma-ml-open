# Run file for Analysis 7
# Aggregate level variables, check against SHAP importance
import datetime as dt
import os

import numpy as np
from Initial_Preprocess import dependent_variable

start = dt.datetime.now()
anal = "Analysis_7"

# ------------------
anal_level = "block"
use_preoptimised_model = True
# ^ has to be true in this script
# todo: clean up this parameter ^
Choose_drop_DV_T1 = True
Test = False
# ------------------

if Choose_drop_DV_T1 == True:
    version = "_without_{}_T1".format(dependent_variable)
    analysis = "{}{}".format(anal, version)
elif Choose_drop_DV_T1 == False:
    version = "_with_{}_T1".format(dependent_variable)
    analysis = "{}{}".format(anal, version)
else:
    version = None
    print("enter either True or False for param Choose_drop_DV_T1")
    breakpoint()

if Test == True:
    t = "_test"
elif Test == False:
    t = ""
else:
    t = None
    print("enter either True or False for param Test")
    breakpoint()

    # enet_cut_off = 2500
    # rf_cut_off = 1500

if use_preoptimised_model == True:
    if anal_level == "block":
        preoptimised_model = 5
        print("Using pre-optimised model {}".format(preoptimised_model))
        if version == "_without_sges_T1":
            start_string = "_26_Aug_2022__14.13"
            enet_cut_off = 0
            rf_cut_off = 0
        #     ^alter
        elif version == "_with_sges_T1":
            start_string = "_24_Aug_2022__20.14"
            enet_cut_off = 0
            rf_cut_off = 0
        else:
            start_string = None
            breakpoint()
    elif anal_level == "indiv":
        preoptimised_model = 6
        print("Using pre-optimised model {}".format(preoptimised_model))
        if version == "_without_sges_T1":
            start_string = "_22_Aug_2022__09.16"
            enet_cut_off = 5000
            rf_cut_off = 1000
        elif version == "_with_sges_T1":
            start_string = "_18_Aug_2022__10.03"
            enet_cut_off = 2700
            rf_cut_off = 700
        else:
            start_string = None
            breakpoint()
    else:
        print("add 'block' or 'indiv' for param anal_level")
        preoptimised_model = None
        breakpoint()

elif use_preoptimised_model == False:
    start_string = start.strftime('_%d_%b_%Y__%H.%M{}'.format(t))
    preoptimised_model = 0
    enet_cut_off = 5000
    rf_cut_off = 1000
else:
    print("add True or False as value for param use_preoptimised_model")
    start_string = None
    preoptimised_model = None
    breakpoint()

if anal_level == "block":
    data_path = "Data/Initial_Preprocess/Data_with_Aggregate_Features/"
    directory_bits = os.fsencode(data_path)
if anal_level == "indiv":
    data_path = "Analysis_6{}/Processed_Data/Processed_Aggregate_Features/".format(version)
    directory_bits = os.fsencode(data_path)

analysis_path = "{}/".format(analysis)
global_seed = np.random.seed(93)
runtime_path = analysis_path + "Outputs/Runtimes/"
runtime_file = "{}_{}.txt".format(analysis, start_string)
perm_cut_off = 0.01
# -----------------------------------------------------------
if __name__ == "__main__":
    print('Running {} (Individual level variables only, no adjustments for multicollinearity)'.format(analysis))

    # SHAP Importance (using data from A6)
    start_perm = dt.datetime.now()
    exec(open("Analysis_7_SHAP_Importance.py").read())
    end_perm = dt.datetime.now()
    perm_runtime = end_perm - start_perm
    print("Permutation importance runtime: {}".format(perm_runtime), file=open(runtime_path + runtime_file, "a"))

    end = dt.datetime.now()
    total = end - start
    print('{} complete. Total time taken: {}'.format(analysis, total), file=open(runtime_path + runtime_file, "a"))
    print('{} complete. Total time taken: {}'.format(analysis, total))