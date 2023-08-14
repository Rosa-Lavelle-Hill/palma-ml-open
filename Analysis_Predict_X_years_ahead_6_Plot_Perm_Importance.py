import os
import re
import pandas as pd
import numpy as np
from functools import reduce
from Functions.Plotting import plot_long, get_custom_palette, heatmap_importance
from Functions.Preprocessing import get_var_names_dict
from RUN_Predict_X_years_ahead_New import start_string, global_seed, t, analysis_path, Choose_drop_DV_T1

# --------
Feature_names_added = True
Include_T5 = False
heat_map = True
# --------

cut_off = 0

test_data_imp_path = analysis_path + "Results/Permutation_Importance/csv/Test_Data/"
directory_bits_test = os.fsencode(test_data_imp_path)

np.random.seed(global_seed)
time_date_string = start_string

Enet_test_var_imp_dict = {}
RF_test_var_imp_dict = {}

# loop through VarImp dfs - test data:

# time_date_string = "_29_Jul_2022__09.54"
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
for df_num in ['1', '2', '3', '4']:
    v_df = all_rf_vars.copy()
    v_df['Time'] = df_num
    v_df = v_df.merge(RF_test_var_imp_dict[df_num], on=['Feature_Block', 'Time'], how='left')
    v_df.fillna(0, inplace=True)
    rf_all_dict[df_num] = v_df
df_long_rf_test_full = pd.concat([v for k, v in rf_all_dict.items()])

enet_all_dict = {}
all_enet_vars = pd.DataFrame(df_long_enet_test['Feature_Block'].unique(), columns=['Feature_Block'])
for df_num in ['1', '2', '3', '4']:
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
    names_dict = get_var_names_dict(var_names)

    # replace names:
    df_long_rf_test_full = df_long_rf_test_full.replace({"Feature_Block": names_dict})
    df_long_enet_test_full = df_long_enet_test_full.replace({"Feature_Block": names_dict})

    # get custom colour palette:
    palette = get_custom_palette(df=var_names, var_name="long_name")

if Feature_names_added == False:
    print("Using old feature names")
    palette = "Paired"

# plot
save_plot_path = analysis_path + "Results/Permutation_Importance/Plots/Over_time/"

df_long_rf_test_full.reset_index(inplace=True)
df_long_rf_test_full.drop(df_long_rf_test_full.loc[df_long_rf_test_full["Time"] == "5"].index, inplace=True)
df_long_enet_test_full.reset_index(inplace=True)
df_long_enet_test_full.drop(df_long_enet_test_full.loc[df_long_enet_test_full["Time"] == "5"].index, inplace=True)

xticks = ["5 --> 6", "5 --> 7", "5 -- >8", "5 --> 9"]
times = [1, 2, 3, 4]

# RF
plot_long(x='Time', y='Importance', data=df_long_rf_test_full, colour="Feature_Block", save_path=save_plot_path,
          save_name="Permutation_Importance_RF_test_set_{}_t{}{}_new".format(time_date_string, cut_off, t),
          xlab="Schooling Year (Grade)", palette= palette,
          ylab="Permutation Feature Importance on Test Data", xticks=xticks,
          title="Random Forest")
# Enet
plot_long(x='Time', y='Importance', data=df_long_enet_test_full, colour="Feature_Block", save_path=save_plot_path,
          save_name="Permutation_Importance_Enet_test_set_{}_t{}{}_new".format(time_date_string, cut_off, t),
          xlab="School year (grade)", palette= palette,
          ylab="Permutation Feature Importance on Test Data", xticks=xticks,
          title="Elastic Net")

# # Plot relative to sges_T1 (rescale using importance of Maths ability Time 1):
#
# if Choose_drop_DV_T1 == False:
#     models_norm_dict = {}
#     df_dict = {'Enet' : df_long_enet_test_full, 'RF' : df_long_rf_test_full }
#     for key, df in df_dict.items():
#         normalised_dict = {}
#         for time in times :
#             time_chunk = df[df["Time"] == str(time)]
#             time_1_imp = time_chunk["Importance"][time_chunk["Feature_Block"] == "Maths Ability Time 1"].values[0]
#             time_chunk.drop(time_chunk.loc[time_chunk['Feature_Block']=="Maths Ability Time 1"].index, inplace=True)
#             time_chunk["Normalised_Importance"] = round(time_chunk["Importance"] / time_1_imp, 4)
#             normalised_dict[time] = time_chunk
#         normalised_df = pd.concat(normalised_dict, axis=0)
#         if Include_T5 == False:
#             normalised_df.drop(normalised_df.loc[normalised_df["Time"] == "5"].index, inplace=True)
#             xticks = ["5 --> 6", "6 --> 7", "7 -- >8", "8 --> 9"]
#         else:
#             continue
#         models_norm_dict[key] = normalised_df
#
#     # Enet
#     var_names = pd.read_csv("Data/MetaData/variable_names.csv")
#     palette = get_custom_palette(df=var_names, var_name="long_name")
#
#     plot_long(x='Time', y='Normalised_Importance', data=models_norm_dict["Enet"], colour="Feature_Block", save_path=save_plot_path,
#                   save_name="Normed_Permutation_Importance_Enet_test_set_{}_t{}{}_new".format(time_date_string, cut_off, t),
#                   xlab="School year (grade)", palette=palette,
#                   ylab="Permutation Feature Importance on Test Data (normalised)", xticks=xticks,
#                   title="Elastic Net")
#     # RF
#     plot_long(x='Time', y='Normalised_Importance', data=models_norm_dict["RF"], colour="Feature_Block", save_path=save_plot_path,
#                   save_name="Normed_Permutation_Importance_RF_test_set_{}_t{}{}_mew".format(time_date_string, cut_off, t),
#                   xlab="School year (grade)", palette=palette,
#                   ylab="Permutation Feature Importance on Test Data (normalised)", xticks=xticks,
#                   title="Random Forest")


if heat_map == True:
    gamma = 0.3

    df_long_enet_test_full.reset_index(inplace=True, drop=True)
    df_long_rf_test_full.reset_index(inplace=True, drop=True)
    df_long_enet_test_full.drop(df_long_enet_test_full.loc[df_long_enet_test_full["Time"] == "5"].index,
                                inplace=True)
    df_long_rf_test_full.drop(df_long_rf_test_full.loc[df_long_rf_test_full["Time"] == "5"].index,
                                inplace=True)

    # Enet
    heatmap_importance(df=df_long_enet_test_full, index="Feature_Block", xticks=xticks,
                       columns="Time", values="Importance", title="Elastic Net", xlab="School year",
                       ylab="Feature Group", save_name="Heatmap_Enet_g{}_{}".format(gamma, time_date_string),
                       save_path=save_plot_path, gamma=gamma)
    # RF
    heatmap_importance(df=df_long_rf_test_full, index="Feature_Block", xticks=xticks,
                       columns="Time", values="Importance", title="Random Forest", xlab="School year",
                       ylab="Feature Group", save_name="Heatmap_RF_g{}_{}".format(gamma, time_date_string),
                       save_path=save_plot_path, gamma=gamma)


print('plot importance done')