import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import joblib
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from Functions.Block_Permutation_Importance import parse_data_to_dict
from Functions.Check import all_vars_in_block
from Functions.Enet_preprocess import dummy_code, remove_special_char_colnames
from Functions.Generic_SHAP import SHAP_enet, SHAP_rf
from Functions.RF_preprocess import to_categorical
from Initial_Preprocess import add_teacher_vars
from Functions.Plotting import plot_perm_box, plot_perm, plot_SHAP, plot_SHAP_over_time, get_custom_palette, plot_long, \
    heatmap_importance
from Functions.Preprocessing import get_var_names_dict, remove_col, shap_dict_to_long_df
from RUN_Predict_same_year import start_string, global_seed, analysis_path, data_save_path, anal_level, directory_bits,\
    Choose_drop_DV_T1, dependent_variable, xticks, times
from Fixed_params import cat_vars, drop_list, block_file, include_T5, changes

shap.initjs()

start_time = dt.datetime.now()
data_path = data_save_path
# ----
only_over_time = False
data_used = 'test'
# ^in this analysis refers to Grade 9 (not 10, so keep in)
heat_map = False

np.random.seed(global_seed)
time_date_string = start_string

# -------
block_dict = parse_data_to_dict(block_file)

rf_param_path = analysis_path + "Results/Best_params/RF/"
enet_param_path = analysis_path + "Results/Best_params/Enet/"
save_plot_path = analysis_path + "Results/Importance/Plots/Test_Data/"

# get custom colour palette:
var_names = pd.read_csv("Data/MetaData/variable_names.csv")
var_names_dict = get_var_names_dict(var_names)
palette = get_custom_palette(df=var_names, var_name="long_name")

fig_save_path_full = analysis_path + "Results/Importance/Plots/Full_Data"
csv_save_path_full = analysis_path + "Results/Importance/csv/Full_Data/"
fig_save_path_test = analysis_path + "Results/Importance/Plots/Test_Data/"
csv_save_path_test = analysis_path + "Results/Importance/csv/Test_Data/"

# -------
df1_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('1', time_date_string))
df2_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('2', time_date_string))
df3_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('3', time_date_string))
df4_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('4', time_date_string))
df5_rf_best_params = joblib.load(rf_param_path+'df{}__{}.pkl'.format('5', time_date_string))

all_params_rf = {'1': df1_rf_best_params, '2': df2_rf_best_params, '3': df3_rf_best_params,
                 '4': df4_rf_best_params, '5': df5_rf_best_params}

df1_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('1', time_date_string))
df2_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('2', time_date_string))
df3_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('3', time_date_string))
df4_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('4', time_date_string))
df5_enet_best_params = joblib.load(enet_param_path+'df{}__{}.pkl'.format('5', time_date_string))

all_params_enet = {'1': df1_enet_best_params, '2': df2_enet_best_params, '3': df3_enet_best_params,
                   '4': df4_enet_best_params, '5': df5_enet_best_params}

csv_save_path = analysis_path + "Results/Importance/csv/"
modelling_data_save = analysis_path + "Modelling_Data/"

rf_shap_dfs_dict = {}
enet_shap_dfs_dict = {}

if only_over_time == False:
    # loop through dfs:
    for file in os.listdir(directory_bits):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            start_df = dt.datetime.now()
            print("--> Processing.... {}".format(filename))
            df_num = [s for s in re.findall(r'\d+', filename)][0]

            df = pd.read_csv(data_save_path+filename)

            # code categorical vars:
            df = to_categorical(df, cat_vars)

            for col in df.columns:
                if col in drop_list:
                    df.drop(col, axis=1, inplace=True)

            if include_T5 == False:
                if df_num == '5':
                    print('skip')
                    continue

            if anal_level == 'block':
                vars_not_in = all_vars_in_block(df, blockdict=block_dict, print=False)

            # define X and y:
            y = df["sges"]
            X = df.drop("sges", axis=1)

            # get best params:
            rf_params = all_params_rf[df_num]
            enet_params = all_params_enet[df_num]

            # Test data (these features less likely to cause over-fitting):
            if data_used == 'test':
                print('Analysing on test data...')

                # ---------------------------------------
                # ENet
                # ---------------------------------------
                model = "Enet"
                shap_dict, names = SHAP_enet(X=X, y=y, df_num=df_num, enet_shap_dfs_dict=enet_shap_dfs_dict,
                                             enet_params=enet_params, modelling_data_save=modelling_data_save,
                                             save_csv_path=analysis_path + "Results/Importance/csv/Test_Data/")

                # plot
                n_features = 10

                plot_type = "bar"
                plot_SHAP(shap_dict, col_list=names.columns.tolist(),
                          n_features=n_features, plot_type=plot_type,
                          save_path= analysis_path + "Results/Importance/Plots/Test_Data/{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                plot_type = "summary"
                plot_SHAP(shap_dict, col_list=names.columns.tolist(),
                          n_features=n_features, plot_type=None,
                          save_path= analysis_path + "Results/Importance/Plots/Test_Data/{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                plot_type = "violin"
                plot_SHAP(shap_dict, col_list=names.columns.tolist(),
                          n_features=n_features, plot_type=plot_type,
                          save_path= analysis_path + "Results/Importance/Plots/Test_Data/{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                # -----------------------------------------
                # RF
                # -----------------------------------------
                model = "RF"
                shap_dict, names = SHAP_rf(X=X, y=y, df_num=df_num, rf_shap_dfs_dict=rf_shap_dfs_dict,
                                             rf_params=rf_params, modelling_data_save=modelling_data_save,
                                             save_csv_path=analysis_path + "Results/Importance/csv/Test_Data/")

                # plot
                n_features = 10

                plot_type = "bar"
                plot_SHAP(shap_dict, col_list=names.columns.tolist(),
                          n_features=n_features, plot_type=plot_type,
                          save_path= analysis_path + "Results/Importance/Plots/Test_Data/{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                plot_type = "summary"
                plot_SHAP(shap_dict, col_list=names.columns.tolist(),
                          n_features=n_features, plot_type=None,
                          save_path= analysis_path + "Results/Importance/Plots/Test_Data/{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                plot_type = "violin"
                plot_SHAP(shap_dict, col_list=names.columns.tolist(),
                          n_features=n_features, plot_type=plot_type,
                          save_path= analysis_path + "Results/Importance/Plots/Test_Data/{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))


if only_over_time == True:
    models = ["Enet", "RF"]
    for model in models:
        imp_data_path = analysis_path + "Results/Importance/csv/Test_Data/{}/".format(model)
        imp_directory_bits = os.fsencode(imp_data_path)
        for file in os.listdir(imp_directory_bits):
            filename = os.fsdecode(file)
            if filename.endswith("SHAP_values.csv"):
                df_num = [s for s in re.findall(r'\d+', filename)][0]
                df = pd.read_csv(imp_data_path + filename, index_col=[0])
                if model == "Enet":
                    enet_shap_dfs_dict[df_num] = df
                if model == "RF":
                    rf_shap_dfs_dict[df_num] = df

#  plot over time
shap_dicts = [enet_shap_dfs_dict, rf_shap_dfs_dict]
for dict in shap_dicts:
    for df_num, df in dict.items():
        df = pd.melt(df)
        # df["Absolute_Average_SHAP"] = df.abs().mean(axis=1)
        df["Time"] = df_num
        df = remove_col(df, "Unnamed: 0")
        dict[df_num] = df

save_path = analysis_path + "Results/Importance/Plots/Over_time/"


if anal_level == "indiv":
    cut_off = 0
    enet_subgroup = shap_dict_to_long_df(dict=enet_shap_dfs_dict, cut_off = cut_off,
                                         csv_save_path=csv_save_path,
                                         csv_save_name = "SHAP_Enet_agg_time_cutoff{}_{}.csv".format(cut_off, start_string))

    save_path = analysis_path + "Results/Importance/Plots/Over_time/"

    if include_T5 == False:
        enet_subgroup.reset_index(inplace=True, drop=True)
        enet_subgroup.drop(enet_subgroup.loc[enet_subgroup["Time"] == "5"].index,
                           inplace=True)

    plot_SHAP_over_time(x="Time", y="Abs_SHAP", data=enet_subgroup,
                        colour="variable", save_path = save_path,
                        save_name = "Enet_abs_sum",
                        xticks = xticks,
                        xlab="Schooling Year (Grade)",
                        ylab="SHAP Importance on Test Data",
                        title = "", size=0.01,
                        fontsize=12,
                        palette=palette)

    # plot average RF
    rf_subgroup = shap_dict_to_long_df(dict=rf_shap_dfs_dict, cut_off=cut_off,
                                       csv_save_path=csv_save_path,
                                       csv_save_name= "SHAP_RF_agg_time.csv")

    if include_T5 == False:
        rf_subgroup.reset_index(inplace=True, drop=True)
        rf_subgroup.drop(rf_subgroup.loc[rf_subgroup["Time"] == "5"].index,
                         inplace=True)

    plot_SHAP_over_time(x="Time", y="Abs_SHAP", data=rf_subgroup,
                        colour="variable", save_path = save_path,
                        save_name = "RF_abs_sum",
                        xticks = xticks,
                        xlab="Schooling Year (Grade)",
                        ylab="SHAP Importance on Test Data",
                        title = "", size=0.01,
                        fontsize=12, palette=palette)

    if heat_map == True:

        cut_off = 1

        enet_subgroup = shap_dict_to_long_df(dict=enet_shap_dfs_dict, cut_off=cut_off,
                                             csv_save_path=csv_save_path,
                                             csv_save_name="SHAP_Enet_agg_time_cutoff{}_{}.csv".format(cut_off, start_string))
        rf_subgroup = shap_dict_to_long_df(dict=rf_shap_dfs_dict, cut_off=cut_off,
                                           csv_save_path=csv_save_path,
                                           csv_save_name="SHAP_RF_agg_time_cutoff{}_{}.csv".format(cut_off, start_string))

        gamma = 0.3

        if include_T5 == False:
            enet_subgroup.reset_index(inplace=True, drop=True)
            rf_subgroup.reset_index(inplace=True, drop=True)
            enet_subgroup.drop(enet_subgroup.loc[enet_subgroup["Time"] == "5"].index,
                                        inplace=True)
            rf_subgroup.drop(rf_subgroup.loc[rf_subgroup["Time"] == "5"].index,
                                        inplace=True)

        # Enet
        # todo: get rid of .'s in col names (double for some reason)
        heatmap_importance(df=enet_subgroup, index="variable", xticks=xticks,
                           columns="Time", values="Abs_SHAP", title="Elastic Net", xlab="School year",
                           ylab="Feature", save_name="Heatmap_indiv_Enet_g{}_{}".format(gamma, time_date_string),
                           save_path=save_path, gamma=gamma, adjust_left=0.4, figsize=(8, 16))
        # RF
        heatmap_importance(df=rf_subgroup, index="variable", xticks=xticks,
                           columns="Time", values="Abs_SHAP", title="Random Forest", xlab="School year",
                           ylab="Feature", save_name="Heatmap_indiv_RF_g{}_{}".format(gamma, time_date_string),
                           save_path=save_path, gamma=gamma, adjust_left=0.4, figsize=(8, 16))

        # ***********play around with different colour options...
        enet_subgroup_2 = enet_subgroup[enet_subgroup.variable != "Maths Ability Time 1"]
        rf_subgroup_2 = rf_subgroup[rf_subgroup.variable != "Maths Ability Time 1"]
        gamma = 0.3
        show_n = 60

        # Enet
        heatmap_importance(df=enet_subgroup_2, index="variable", xticks=xticks, sort_by="overall", fontsize=12, tick_font_size=7,
                           columns="Time", values="Abs_SHAP", title="Elastic Net", xlab="School year", show_n=show_n,
                           ylab="Feature", save_name="Heatmap_indiv_Enet_g{}_{}_new".format(gamma, time_date_string),
                           save_path=save_path, gamma=gamma, adjust_left=0.4, figsize=(8, 16))
        # RF
        heatmap_importance(df=rf_subgroup_2, index="variable", xticks=xticks, sort_by="overall", fontsize=12, tick_font_size=7,
                           columns="Time", values="Abs_SHAP", title="Random Forest", xlab="School year", show_n=show_n,
                           ylab="Feature", save_name="Heatmap_indiv_RF_g{}_{}_new".format(gamma, time_date_string),
                           save_path=save_path, gamma=gamma, adjust_left=0.4, figsize=(8, 16))

# todo: then plot variance, and pos/neg for each variable
# =========================
# At group-level
# =========================
if anal_level == "block":

    block_dict = parse_data_to_dict(block_file)

    # Enet
    cut_off = 0
    enet_subgroup_agg = shap_dict_to_long_df(dict=enet_shap_dfs_dict, cut_off = cut_off,
                                         csv_save_path=csv_save_path,
                                         csv_save_name = "SHAP_Enet_agg_time_{}.csv".format(cut_off))

    enet_full = enet_subgroup_agg.merge(var_names, left_on="variable", right_on="long_name")\
        .drop("long_name", axis=1)
    enet_full["block"] = 0
    for index, row in enet_full.iterrows():
        for key, item in block_dict.items():
            sh = row["short_name"]
            sh = sh.split('(')[0]
            sh = sh.strip()
            sh = sh.replace(' ', "_")
            if sh in item:
                enet_full.loc[index, "block"] = key
    enet_full = enet_full[enet_full["block"]!=0]
    enet_full.sort_values(by=['Time', "variable"], inplace=True)
    enet_group = enet_full[['Time', "block", "Abs_SHAP"]].groupby(['Time', "block"]).sum().reset_index()

    # rename blocks:
    enet_group.replace({"block": var_names_dict}, inplace=True)
    enet_group.sort_values(by="Abs_SHAP", ascending=False, inplace=True)
    csv_block_save = analysis_path + "Results/Importance/csv/Test_Data/Enet/block/"
    enet_group.to_csv(csv_block_save + "sum_importance_over_time.csv")

    if include_T5 == False:
        enet_group.reset_index(inplace=True, drop=True)
        enet_group.drop(enet_group.loc[enet_group["Time"] == "5"].index,
                           inplace=True)
        t5 = "_without_t5"
    else:
        t5 = ""

    plot_SHAP_over_time(x="Time", y="Abs_SHAP", data=enet_group,
                        colour="block", save_path = save_path,
                        save_name = "Enet_block_abs_sum{}{}{}".format(start_string, t5, changes),
                        xticks = xticks,
                        xlab="Schooling Year (Grade)",
                        ylab="SHAP Importance on Test Data",
                        title = "Elastic Net", size=0.01,
                        fontsize=12,
                        palette=palette)
    # RF
    rf_subgroup_agg = shap_dict_to_long_df(dict=rf_shap_dfs_dict, cut_off = cut_off,
                                         csv_save_path=csv_save_path,
                                         csv_save_name = "SHAP_RF_agg_time_{}.csv".format(cut_off))

    rf_full = rf_subgroup_agg.merge(var_names, left_on="variable", right_on="long_name")\
        .drop("long_name", axis=1)
    rf_full["block"] = 0
    for index, row in rf_full.iterrows():
        for key, item in block_dict.items():
            sh = row["short_name"]
            sh = sh.split('(')[0]
            sh = sh.strip()
            sh = sh.replace(' ', "_")
            if sh in item:
                rf_full.loc[index, "block"] = key
    rf_full = rf_full[rf_full["block"]!=0]
    rf_full.sort_values(by=['Time', "variable"], inplace=True)
    rf_group = rf_full[['Time', "block", "Abs_SHAP"]].groupby(['Time', "block"]).sum().reset_index()

    # rename blocks:
    rf_group.replace({"block": var_names_dict}, inplace=True)
    rf_group.sort_values(by="Abs_SHAP", ascending=False, inplace=True)
    csv_block_save = analysis_path + "Results/Importance/csv/Test_Data/RF/block/"
    rf_group.to_csv(csv_block_save + "sum_importance_over_time.csv")

    if include_T5 == False:
        rf_group.reset_index(inplace=True, drop=True)
        rf_group.drop(rf_group.loc[rf_group["Time"] == "5"].index,
                         inplace=True)
        t5 = "_without_t5"
    else:
        t5 = ""

    plot_SHAP_over_time(x="Time", y="Abs_SHAP", data=rf_group,
                        colour="block", save_path = save_path,
                        save_name = "RF_block_abs_sum{}{}{}".format(start_string, t5, changes),
                        xticks = xticks,
                        xlab="Schooling Year (Grade)",
                        ylab="SHAP Importance on Test Data",
                        title = "Random Forest", size=0.01,
                        fontsize=12,
                        palette=palette)

    # ===========================================================================

    # heatmaps
    if heat_map == True:
        gamma = 0.3

        if include_T5 == False:
            enet_group.reset_index(inplace=True, drop=True)
            rf_group.reset_index(inplace=True, drop=True)
            enet_group.drop(enet_group.loc[enet_group["Time"] == "5"].index,
                                        inplace=True)
            rf_group.drop(rf_group.loc[rf_group["Time"] == "5"].index,
                                        inplace=True)

        # Enet
        heatmap_importance(df=enet_group, index="block", xticks=xticks,
                           columns="Time", values="Abs_SHAP", title="Elastic Net", xlab="School year",
                           ylab="Feature Group", save_name="SHAP_block_Heatmap_Enet_g{}_{}{}".format(gamma, time_date_string, changes),
                           save_path=save_path, gamma=gamma)
        # RF
        heatmap_importance(df=rf_group, index="block", xticks=xticks,
                           columns="Time", values="Abs_SHAP", title="Random Forest", xlab="School year",
                           ylab="Feature Group", save_name="SHAP_block_Heatmap_RF_g{}_{}{}".format(gamma, time_date_string, changes),
                           save_path=save_path, gamma=gamma)
# =========================
end_time = dt.datetime.now()
run_time = end_time - start_time
print("SHAP importance runtime: {}".format(run_time))

print('done!')