import os
import re
import pandas as pd
import numpy as np
from Functions.Plotting import plot_long, get_custom_palette
from Functions.Preprocessing import get_var_names_dict_agg, remove_col
from RUN_Analysis_6 import start_string, global_seed, perm_cut_off, analysis_path

# --------
Feature_names_added = True
# --------
# test params: ***comment out
# analysis_path = "Analysis_6_without_sges_T1/"
# time_date_string = "_26_Aug_2022__14.14"
time_date_string = start_string
# ^^comment in

test_data_imp_path = analysis_path + "Results/Importance/csv/Test_Data/"
directory_bits_test = os.fsencode(test_data_imp_path)

np.random.seed(global_seed)

Enet_test_var_imp_dict = {}
RF_test_var_imp_dict = {}

cut_off = perm_cut_off
# loop through VarImp dfs - test data:

for file in os.listdir(directory_bits_test):
    filename = os.fsdecode(file)
    if filename.endswith(time_date_string+".csv"):
        df_num = filename[2]
        if 'Enet' in filename:
            print("--> Processing.... {}".format(filename))

            # test data importance
            test_imp_df = pd.read_csv(test_data_imp_path + filename)
            test_imp_df = remove_col(df=test_imp_df, drop_col="Unnamed: 0")

            # quick_fix:
            test_imp_df.drop(test_imp_df[test_imp_df["Feature"] == "School_Class"].index, inplace = True)

            # standardise names across times
            for var in test_imp_df['Feature']:
                if var == 'gma_jz4':
                    test_imp_df.loc[test_imp_df["Feature"] == var, ["Feature"]] = "Primary School Maths Grade"
                    var = "End 1st School Maths Grade"
                if var.startswith("gma_jz"):
                    test_imp_df.loc[test_imp_df["Feature"] == var, ["Feature"]] = "Teacher's Maths Grade"
                    var = "Last Year's Maths Grade"

            test_imp_df.loc[test_imp_df['Importance']>1, 'Importance'] = 0

            test_imp_df['Time'] = df_num
            test_imp_df = test_imp_df[test_imp_df["Importance"] > cut_off]
            Enet_test_var_imp_dict[df_num] = test_imp_df

        if 'RF' in filename:
            print("--> Processing.... {}".format(filename))
            df_num = [s for s in re.findall(r'\d+', filename)][0]

            # test data importance
            test_imp_df = pd.read_csv(test_data_imp_path + filename)
            test_imp_df.drop("Unnamed: 0", axis=1, inplace=True)

            # quick_fix:
            test_imp_df.drop(test_imp_df[test_imp_df["Feature"] == "School_Class"].index, inplace=True)

            # standardise names across times
            for var in test_imp_df['Feature']:
                if var == 'gma_jz4':
                    test_imp_df.loc[test_imp_df["Feature"] == var, ["Feature"]] = "Primary School Maths Grade"
                    var = "End 1st School Maths Grade"
                if var.startswith("gma_jz"):
                    test_imp_df.loc[test_imp_df["Feature"] == var, ["Feature"]] = "Teacher's Maths Grade"
                    var = "Last Year's Maths Grade"

            test_imp_df['Time'] = df_num
            test_imp_df = test_imp_df[test_imp_df["Importance"] > cut_off]
            RF_test_var_imp_dict[df_num] = test_imp_df

df_long_rf_test = pd.concat([v for k, v in RF_test_var_imp_dict.items()])
df_long_enet_test = pd.concat([v for k, v in Enet_test_var_imp_dict.items()])

# fill in zeros:
rf_all_dict = {}
all_rf_vars = pd.DataFrame(df_long_rf_test['Feature'].unique(), columns=['Feature'])
for df_num in ['1', '2', '3', '4', '5']:
    v_df = all_rf_vars.copy()
    v_df['Time'] = df_num
    v_df = v_df.merge(RF_test_var_imp_dict[df_num], on=['Feature', 'Time'], how='left')
    v_df.fillna(0, inplace=True)
    rf_all_dict[df_num] = v_df
df_long_rf_test_full = pd.concat([v for k, v in rf_all_dict.items()])

enet_all_dict = {}
all_enet_vars = pd.DataFrame(df_long_enet_test['Feature'].unique(), columns=['Feature'])
for df_num in ['1', '2', '3', '4', '5']:
    v_df = all_enet_vars.copy()
    v_df['Time'] = df_num
    v_df = v_df.merge(Enet_test_var_imp_dict[df_num], on=['Feature', 'Time'], how='left')
    v_df.fillna(0, inplace=True)
    enet_all_dict[df_num] = v_df
df_long_enet_test_full = pd.concat([v for k, v in enet_all_dict.items()])

# get all variables used for translation:
all_vars = pd.DataFrame(pd.concat([all_rf_vars, all_enet_vars], axis=0)['Feature'].unique())
all_vars.to_csv(analysis_path + "Outputs/Feature_Interpretation/ImportantFeatures/Raw__{}.csv".format(time_date_string))

if Feature_names_added == True:
    "Loading variable names"
    # load new names:
    var_names = pd.read_csv("Data/MetaData/variable_names.csv")

    names_dict = get_var_names_dict_agg(var_names)
    var_names_agg = pd.DataFrame.from_dict(names_dict, orient='index').reset_index()
    var_names_agg.columns = ["short_name", "long_name"]

    # replace names:
    df_long_rf_test_full = df_long_rf_test_full.replace({"Feature": names_dict})
    df_long_enet_test_full = df_long_enet_test_full.replace({"Feature": names_dict})

    # get custom colour palette:
    palette = get_custom_palette(df=var_names_agg, var_name="long_name")

elif Feature_names_added == False:
    print("Using old feature names")
    palette = "Paired"

# plot
save_plot_path = analysis_path + "Results/Importance/Plots/Over_time/"
xticks = ["5 --> 6", "6 --> 7", "7 --> 8", "8 --> 9", "9 --> 10"]
# RF
plot_long(x='Time', y='Importance', data=df_long_rf_test_full, colour="Feature", save_path=save_plot_path,
          save_name="Permutation_Importance_RF_test_set_{}_t{}".format(time_date_string, cut_off),
          xlab="Schooling Year (Grade)", palette=palette,
          ylab="Permuation Feature Importance on test data (0-1)", xticks=xticks,
          title=" ".format(cut_off))
# Enet
plot_long(x='Time', y='Importance', data=df_long_enet_test_full, colour="Feature", save_path=save_plot_path,
          save_name="Permutation_Importance_Enet_test_set_{}_t{}".format(time_date_string, cut_off),
          xlab="School year (grade)", palette=palette,
          ylab="Permuation Feature Importance on test data (0-1)", xticks=xticks,
          title=" ".format(cut_off))

print('plot importance done')