import os
import re
import pandas as pd
import numpy as np
from functools import reduce
from Initial_Preprocess import add_teacher_vars
from Functions.Block_Permutation_Importance import parse_data_to_dict
from Functions.Plotting import plot_long, get_custom_palette, heatmap_importance
from Functions.Preprocessing import get_var_names_dict
from RUN_Analysis_4 import start_string, global_seed, t, analysis_path
from Fixed_params import block_file

changes = "track_changes"

data_path = "Data/Initial_Preprocess/"
directory_bits = os.fsencode("Data/Initial_Preprocess/")
version = "_with_sges_T1"
cor_save_path = "Analysis_4_with_sges_T1/Outputs/Block_correlations/"

block_df = pd.read_csv(block_file)
block_dict = parse_data_to_dict(block_file)

var_names = pd.read_csv("Data/MetaData/variable_names.csv")
names_dict = get_var_names_dict(var_names)

for file in os.listdir(directory_bits):
    filename = os.fsdecode(file)
    if filename.endswith("_preprocessed{}.csv".format(version)):
        print("--> Processing.... {}".format(filename))
        df_num = [s for s in re.findall(r'\d+', filename)][0]

        df = pd.read_csv(data_path+filename)

        # reorder df cols
        block_vars = []
        for block_name, all_block_vars in block_dict.items():
            for var in all_block_vars:
                if var in df.columns.to_list():
                    block_vars.append(var)

        df_ordered = df[block_vars]

        # add nice names
        df_ordered.rename(names_dict, inplace=True, axis=1)

        # create cor matrix
        cor = round(df_ordered.corr(), 2)
        cor.to_csv(cor_save_path+"df_{}{}.csv".format(df_num, changes))

        # plot and save

print('done!')