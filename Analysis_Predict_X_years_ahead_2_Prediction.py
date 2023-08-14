import os

import numpy as np
import datetime as dt

print("running prediction file")

from RUN_Predict_X_years_ahead_New import start_string, global_seed, Additional_pred_checks_only,\
    rf_params, enet_params, analysis_path, version, final_data_path, directory_bits, Additional_pred_checks,\
    Test, anal_level
if anal_level == "block":
    from RUN_Predict_X_years_ahead_New import block_file
from Functions.Generic_Prediction import predict

start = dt.datetime.now()
np.random.seed(global_seed)

if __name__ == "__main__":
    # load variable groups

    if Additional_pred_checks_only == False:

        predict(directory_bits=directory_bits, analysis_path=analysis_path,
                data_path=final_data_path, block_file=block_file, test=Test,
                version=version, start_string=start_string, enet_params=enet_params,
                rf_params=rf_params)

        end_time = dt.datetime.now()
        run_time = end_time - start
        print('done! run time: {}'.format(run_time))

    if Additional_pred_checks == True:
        start_string = start_string + "_add"
        data_path = analysis_path + "Processed_Data/Additional_pred_ahead/"
        directory_bits = os.fsencode(data_path)
        predict(directory_bits=directory_bits, analysis_path=analysis_path,
                data_path=data_path, addition_pred_checks=True, block_file=block_file, test=Test,
                version=version, start_string=start_string, enet_params=enet_params,
                rf_params=rf_params)
