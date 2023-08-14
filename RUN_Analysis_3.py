# Run file for Analysis 3
# Individual level variables, check against a RF (with surrogates)/ SHAP importance
import datetime as dt
import os

import numpy as np

from Functions.Set_params import set_params_do_initial_preprocess
from Initial_Preprocess import preprocess_drop_DV_T1, dependent_variable

start = dt.datetime.now()
anal = "Analysis_3"

anal_level = "block"
Do_initial_preprocess = False
Test = False
use_preoptimised_model = True
force_use_model_2 = True
Choose_drop_DV_T1 = False

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
    if anal_level == "block":
        if force_use_model_2 == True:
            preoptimised_model = 2
        else:
            preoptimised_model = 1
        print("Using a pre-optimised model {}".format(preoptimised_model))
        if version == "_without_sges_T1":
            start_string = "_26_Aug_2022__14.09"
        if version == "_with_sges_T1":
            # start_string = "_29_Jul_2022__09.54"
            # start_string = "_04_Nov_2022__09.05"
            # start_string = "_30_Nov_2022__17.22"
            # start_string = "_02_Jan_2023__12.07"
            # start_string = "_14_Jan_2023__21.19"
            start_string = "_25_Jan_2023__12.15"
    if anal_level == "indiv":
        preoptimised_model = 2
        print("Using a pre-optimised model {}".format(preoptimised_model))
        if version == "_without_sges_T1":
            start_string = "_11_Aug_2022__16.01"
        if version == "_with_sges_T1":
            # start_string = "_29_Jul_2022__09.54"
            # start_string = "_04_Nov_2022__09.05"
            # start_string = "_30_Nov_2022__17.22"
            # start_string = "_02_Jan_2023__12.07"
            # start_string = "_14_Jan_2023__21.19"
            start_string = "_25_Jan_2023__12.15"
if use_preoptimised_model == False:
    preoptimised_model = 3
    start_string = start.strftime('_%d_%b_%Y__%H.%M{}'.format(t))


data_path = "Analysis_2{}/Processed_Data/".format(version)
directory_bits = os.fsencode(data_path)

analysis_path = "{}/".format(analysis)
global_seed = np.random.seed(93)
runtime_path = analysis_path + "Outputs/Runtimes/"
runtime_file = "{}_{}.txt".format(analysis, start_string)
perm_cut_off = 0.01
# -----------------------------------------------------------
if __name__ == "__main__":
    print('Running {} (Individual level variables only, no adjustments for multicollinearity)'.format(analysis))

    # Preprocess
    if Do_initial_preprocess == True:
        exec(open("Initial_Preprocess.py").read())
        end_preprocess = dt.datetime.now()
        preprocess_runtime = end_preprocess - start
        print("Preprocess runtime: {}".format(preprocess_runtime), file=open(runtime_path + runtime_file, "w"))

    # SHAP Importance
    start_perm = dt.datetime.now()
    exec(open("Analysis_3c_SHAP_Importance.py").read())
    end_perm = dt.datetime.now()
    perm_runtime = end_perm - start_perm
    print("Permutation importance runtime: {}".format(perm_runtime), file=open(runtime_path + runtime_file, "a"))

    end = dt.datetime.now()
    total = end - start
    print('{} complete. Total time taken: {}'.format(analysis, total), file=open(runtime_path + runtime_file, "a"))
    print('{} complete. Total time taken: {}'.format(analysis, total))
