import os
import re
import pandas as pd
import numpy as np
from functools import reduce
from Functions.Plotting import plot_long, get_custom_palette
from Functions.Preprocessing import get_var_names_dict_agg
from RUN_Analysis_8 import start_string, global_seed, t, analysis_path, perm_cut_off

# --------
Feature_names_added = True
# --------
cut_off = perm_cut_off

test_data_imp_path = analysis_path + "Results/Importance/csv/Test_Data/"
directory_bits_test = os.fsencode(test_data_imp_path)

np.random.seed(global_seed)
time_date_string = start_string

Enet_test_var_imp_dict = {}
RF_test_var_imp_dict = {}

# loop through VarImp dfs - test data:

# time_date_string = "_24_Aug_2022__20.10"
# ^test run string
print("Running plotting of block permutation importance analysis")

for file in os.listdir(directory_bits_test):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        df_num = re.findall(r'\d+', filename)[0]
        if 'Enet' in filename:
            print("--> Processing.... {}".format(filename))

            # test data importance
            test_imp_df = pd.read_csv(test_data_imp_path + filename)
            test_imp_df.drop("Unnamed: 0", axis=1, inplace=True)

            # standardise names across times
            for var in test_imp_df['Feature_Block']:
                if var == 'gma_jz4':
                    test_imp_df.loc[test_imp_df["Feature_Block"] == var, ["Feature_Block"]] = "Primary School Maths Grade"
                    var = "End 1st School Maths Grade"
                if var.startswith("gma_jz"):
                    test_imp_df.loc[test_imp_df["Feature_Block"] == var, ["Feature_Block"]] = "Teacher's Maths Grade"
                    var = "Last Year's Maths Grade"

            test_imp_df['Time'] = df_num
            test_imp_df = test_imp_df[test_imp_df["Importance"] > cut_off]
            Enet_test_var_imp_dict[df_num] = test_imp_df

        if 'RF' in filename:
            print("--> Processing.... {}".format(filename))
            df_num = [s for s in re.findall(r'\d+', filename)][0]

            # test data importance
            test_imp_df = pd.read_csv(test_data_imp_path + filename)
            test_imp_df.drop("Unnamed: 0", axis=1, inplace=True)

            # standardise names across times
            for var in test_imp_df['Feature_Block']:
                if var == 'gma_jz4':
                    test_imp_df.loc[test_imp_df["Feature_Block"] == var, ["Feature_Block"]] = "Primary School Maths Grade"
                    var = "End 1st School Maths Grade"
                if var.startswith("gma_jz"):
                    test_imp_df.loc[test_imp_df["Feature_Block"] == var, ["Feature_Block"]] = "Teacher's Maths Grade"
                    var = "Last Year's Maths Grade"

            test_imp_df['Time'] = df_num
            test_imp_df = test_imp_df[test_imp_df["Importance"] > cut_off]
            RF_test_var_imp_dict[df_num] = test_imp_df

df_long_rf_test = pd.concat([v for k, v in RF_test_var_imp_dict.items()])
df_long_enet_test = pd.concat([v for k, v in Enet_test_var_imp_dict.items()])

# fill in zeros:
rf_all_dict = {}
all_rf_vars = pd.DataFrame(df_long_rf_test['Feature_Block'].unique(), columns=['Feature_Block'])
for df_num in ['1', '2', '3', '4', '5']:
    v_df = all_rf_vars.copy()
    v_df['Time'] = df_num
    v_df = v_df.merge(RF_test_var_imp_dict[df_num], on=['Feature_Block', 'Time'], how='left')
    v_df.fillna(0, inplace=True)
    rf_all_dict[df_num] = v_df
df_long_rf_test_full = pd.concat([v for k, v in rf_all_dict.items()])

enet_all_dict = {}
all_enet_vars = pd.DataFrame(df_long_enet_test['Feature_Block'].unique(), columns=['Feature_Block'])
for df_num in ['1', '2', '3', '4', '5']:
    v_df = all_enet_vars.copy()
    v_df['Time'] = df_num
    v_df = v_df.merge(Enet_test_var_imp_dict[df_num], on=['Feature_Block', 'Time'], how='left')
    v_df.fillna(0, inplace=True)
    enet_all_dict[df_num] = v_df
df_long_enet_test_full = pd.concat([v for k, v in enet_all_dict.items()])

if Feature_names_added == True:
    print("Loading variable names")
    # load new names:
    var_names = pd.read_csv("Data/MetaData/variable_names.csv")
    names_dict = get_var_names_dict_agg(var_names)
    var_names_agg = pd.DataFrame.from_dict(names_dict, orient='index').reset_index()
    var_names_agg.columns = ["short_name", "long_name"]

    # replace names:
    df_long_rf_test_full = df_long_rf_test_full.replace({"Feature_Block": names_dict})
    df_long_enet_test_full = df_long_enet_test_full.replace({"Feature_Block": names_dict})

    # get custom colour palette:
    palette = get_custom_palette(df=var_names_agg, var_name="long_name")

if Feature_names_added == False:
    print("Using old feature names")
    palette = "Paired"

# plot
save_plot_path = analysis_path + "Results/Importance/Plots/Over_time/"
xticks = ["5 --> 6", "6 --> 7", "7 -- >8", "8 --> 9", "9 --> 10"]
# RF
plot_long(x='Time', y='Importance', data=df_long_rf_test_full, colour="Feature_Block", save_path=save_plot_path,
          save_name="Permutation_Importance_RF_test_set_{}_t{}{}".format(time_date_string, cut_off, t),
          xlab="Schooling Year (Grade)", palette=palette,
          ylab="Permuation Feature Importance on test data (0-1)", xticks=xticks,
          title=" ")
# Enet
plot_long(x='Time', y='Importance', data=df_long_enet_test_full, colour="Feature_Block", save_path=save_plot_path,
          save_name="Permutation_Importance_Enet_test_set_{}_t{}{}".format(time_date_string, cut_off, t),
          xlab="School year (grade)", palette=palette,
          ylab="Permuation Feature Importance on test data (0-1)", xticks=xticks,
          title=" ")



print('plot importance done')