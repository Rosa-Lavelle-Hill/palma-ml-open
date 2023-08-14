
import os
import re
import pandas as pd
import numpy as np
import datetime as dt
from Functions.RF_preprocess import to_categorical
from RUN_Predict_same_year import original_data_load_path, data_save_path
from Fixed_params import cat_vars, drop_list

old_dependent_variable = "sges"

# set save paths and directories
directory_bits = os.fsencode(original_data_load_path)

if __name__ == "__main__":
    decimal_places = 4
    dv_T1 = old_dependent_variable+'_T1'

    dfs_dict = {}
    # loop through dfs:
    for file in os.listdir(directory_bits):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            start_df = dt.datetime.now()
            print("--> Processing.... {}".format(filename))
            df_num = [s for s in re.findall(r'\d+', filename)][0]

            df = pd.read_csv(original_data_load_path+filename)

            df.drop(old_dependent_variable, inplace=True, axis=1)

            # save
            df.to_csv(data_save_path+"predict_same_year_df{}.csv".format(str(df_num)))

print('data created!')


