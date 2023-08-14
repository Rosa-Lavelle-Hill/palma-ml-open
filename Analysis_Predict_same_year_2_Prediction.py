import numpy as np
import datetime as dt

print("running prediction file")

from RUN_Predict_same_year import start_string, global_seed, dependent_variable, \
    rf_params, enet_params, analysis_path, data_save_path, directory_bits, Compare_DV_T1,\
    Test
from Functions.Generic_Prediction import predict
from Fixed_params import block_file


start = dt.datetime.now()
np.random.seed(global_seed)

if __name__ == "__main__":
    print("using predict function")

    predict(directory_bits=directory_bits, analysis_path=analysis_path,
            data_path=data_save_path, block_file=block_file, test=Test,
            start_string=start_string, enet_params=enet_params, compare_DV_T1_baseline=Compare_DV_T1,
            rf_params=rf_params, dependent_variable=dependent_variable)

    end_time = dt.datetime.now()
    run_time = end_time - start
    print('done! run time: {}'.format(run_time))