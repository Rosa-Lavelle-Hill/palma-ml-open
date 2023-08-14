
import os
import numpy as np
import datetime as dt

print("running prediction file")

from RUN_Analysis_2_New import start_string, global_seed, rf_params, enet_params,\
    analysis_path, version, final_data_path, directory_bits, Test, add_teacher_vars, change_cog, change_emotions
from Functions.Generic_Prediction import predict
from Fixed_params import block_file

start = dt.datetime.now()
np.random.seed(global_seed)

if __name__ == "__main__":

    predict(directory_bits=directory_bits, analysis_path=analysis_path,
            data_path=final_data_path, block_file=block_file, test=Test,
            version=version, start_string=start_string, enet_params=enet_params,
            rf_params=rf_params)

    end_time = dt.datetime.now()
    run_time = end_time - start
    print('done! run time: {}'.format(run_time))

end_time = dt.datetime.now()
run_time = end_time - start
print('done! run time: {}'.format(run_time))