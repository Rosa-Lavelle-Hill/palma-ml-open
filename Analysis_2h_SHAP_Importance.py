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
from Functions.Enet_preprocess import dummy_code, remove_special_char_colnames
from Functions.RF_preprocess import to_categorical
from Initial_Preprocess import add_teacher_vars
from Functions.Plotting import plot_perm_box, plot_perm, plot_SHAP, plot_SHAP_over_time, get_custom_palette, plot_long, \
    heatmap_importance
from Functions.Preprocessing import get_var_names_dict, remove_col, shap_dict_to_long_df
from RUN_Analysis_2 import start_string, global_seed, analysis_path, version, Choose_drop_DV_T1
from Fixed_params import cat_vars, drop_list, block_file
from Analysis_2g_Predict_x_years_ahead import new_data_path, dependent_variable

shap.initjs()
data_path = new_data_path
directory_bits = os.fsencode(data_path)
preoptimised_model = 2

start_time = dt.datetime.now()
# ----
only_over_time = False
data_used = 'test'
Include_T5 = False
heat_map = False
anal_level = "indiv"
changes = "track_changes"

np.random.seed(global_seed)
time_date_string = start_string

if Include_T5 == False:
    times = [1, 2, 3, 4]
    xticks = ["5 --> 6", "6 --> 7", "7 -- >8", "8 --> 9"]
else:
    times = [1, 2, 3, 4, 5]
    xticks = ["5 --> 6", "6 --> 7", "7 -- >8", "8 --> 9", "9 --> 10"]

# -------
load_analysis_path = "Analysis_{}{}/".format(preoptimised_model, version)
rf_param_path = load_analysis_path + "Results/Best_params/RF/"
enet_param_path = load_analysis_path + "Results/Best_params/Enet/"
save_plot_path = analysis_path + "Results/Predict_ahead_x_years/Importance/Plots/Test_Data/"

# get custom colour palette:
var_names = pd.read_csv("Data/MetaData/variable_names.csv")
var_names_dict = get_var_names_dict(var_names)
palette = get_custom_palette(df=var_names, var_name="long_name")

fig_save_path_full = analysis_path + "Results/Predict_ahead_x_years/Importance/Plots/Full_Data"
csv_save_path_full = analysis_path + "Results/Predict_ahead_x_years/Importance/csv/Full_Data/"
fig_save_path_test = analysis_path + "Results/Predict_ahead_x_years/Importance/Plots/Test_Data/"
csv_save_path_test = analysis_path + "Results/Predict_ahead_x_years/Importance/csv/Test_Data/"

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

csv_save_path = analysis_path + "Results/Predict_ahead_x_years/Importance/"
modelling_data_save = analysis_path + "Results/Predict_ahead_x_years/Modelling_Data/"

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

            df = pd.read_csv(data_path+filename)

            # check removed rows where no y
            df.dropna(axis=0, inplace=True, subset=[dependent_variable])

            # code categorical vars:
            df = to_categorical(df, cat_vars)

            for col in df.columns:
                if col in drop_list:
                    df.drop(col, axis=1, inplace=True)

            # pipeline steps:
            imputer = IterativeImputer(missing_values=np.nan, max_iter=10, random_state=93)
            transformer = StandardScaler()

            # additional preprocess steps for Enet:
            df_enet = dummy_code(df=df, vars_to_code=cat_vars)
            df_enet = remove_special_char_colnames(df_enet, '.')

            # define X and y:
            y_enet = df_enet["sges"]
            X_enet = df_enet.drop("sges", axis=1)

            y_rf = df["sges"]
            X_rf = df.drop("sges", axis=1)

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
                # split data using same params (should be same as the train/test set used for prediction)
                X_train, X_test, y_train, y_test = train_test_split(X_enet, y_enet, random_state=93, test_size=0.2, shuffle=True)
                print("y train n={}; mean={},\ny test n={}; mean={}".format(y_train.shape[0], round(y_train.mean(), 2),
                                                                            y_test.shape[0], round(y_test.mean(), 2)))

                # save data sets:
                X_train.to_csv(modelling_data_save + "{}/df{}_X_train.csv".format(model, df_num))
                X_test.to_csv(modelling_data_save + "{}/df{}_X_test.csv".format(model, df_num))
                pd.DataFrame(y_train).to_csv(modelling_data_save + "{}/df{}_y_train.csv".format(model, df_num))
                pd.DataFrame(y_test).to_csv(modelling_data_save + "{}/df{}_y_test.csv".format(model, df_num))

                print('Fitting Elastic Net...')

                elastic_net_regression = ElasticNet(alpha=enet_params['regression__alpha'],
                                                    l1_ratio=enet_params['regression__l1_ratio'],
                                                    max_iter=enet_params['regression__max_iter'],
                                                    tol= enet_params['regression__tol'])

                # fit each step of the pipeline on train data, transform test data:

                # fit to and transform train
                X_train= imputer.fit_transform(X_train)
                X_train = transformer.fit_transform(X_train)
                elastic_net_regression.fit(X_train, y_train)

                # transform test
                X_test = imputer.transform(X_test)
                X_test = transformer.transform(X_test)

                # Fit the explainer
                explainer = shap.LinearExplainer(elastic_net_regression, X_test, feature_pertubation="correlation_dependent")

                # Calculate the SHAP values and save
                shap_dict = explainer(X_test)
                shap_values = explainer.shap_values(X_test)
                X_names = X_enet.rename(columns=var_names_dict)
                shap_values_df = pd.DataFrame(shap_values, columns=X_names.columns.tolist())
                save_csv_path = analysis_path + "Results/Predict_ahead_x_years/Importance/csv/Test/{}/".format(model)
                shap_values_df.to_csv(save_csv_path + "df{}_SHAP_values.csv".format(df_num))
                enet_shap_dfs_dict[df_num] = shap_values_df

                # plot
                n_features = 10

                plot_type = "bar"
                plot_SHAP(shap_dict, col_list=X_names.columns.tolist(),
                          n_features=n_features, plot_type=plot_type,
                          save_path= save_plot_path + "{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                plot_type = "summary"
                plot_SHAP(shap_dict, col_list=X_names.columns.tolist(),
                          n_features=n_features, plot_type=None,
                          save_path= save_plot_path + "{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                plot_type = "violin"
                plot_SHAP(shap_dict, col_list=X_names.columns.tolist(),
                          n_features=n_features, plot_type=plot_type,
                          save_path= save_plot_path + "{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                # -----------------------------------------
                # RF
                # -----------------------------------------
                model = "RF"

                X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, random_state=93, test_size=0.2,
                                                                    shuffle=True)
                print("y train n={}; mean={},\ny test n={}; mean={}".format(y_train.shape[0], round(y_train.mean(), 2),
                                                                            y_test.shape[0], round(y_test.mean(), 2)))
                # save data sets:
                X_train.to_csv(modelling_data_save + "{}/df{}_X_train.csv".format(model, df_num))
                X_test.to_csv(modelling_data_save + "{}/df{}_X_test.csv".format(model, df_num))
                pd.DataFrame(y_train).to_csv(modelling_data_save + "{}/df{}_y_train.csv".format(model, df_num))
                pd.DataFrame(y_test).to_csv(modelling_data_save + "{}/df{}_y_test.csv".format(model, df_num))

                random_forest_regression = RandomForestRegressor(max_depth=rf_params['regression__max_depth'],
                                                    max_features=rf_params['regression__max_features'],
                                                    min_samples_split=rf_params['regression__min_samples_split'],
                                                    random_state=rf_params['regression__random_state'],
                                                    n_estimators= rf_params['regression__n_estimators'])

                print('Fitting Random Forest...')

                # fit to and transform train
                X_train= imputer.fit_transform(X_train)
                X_train = transformer.fit_transform(X_train)
                random_forest_regression.fit(X_train, y_train)

                # transform test
                X_test = imputer.transform(X_test)
                X_test = transformer.transform(X_test)

                # Fit the explainer
                explainer = shap.TreeExplainer(random_forest_regression, X_test, feature_pertubation="tree_path_dependent")

                # Calculate the SHAP values and save
                shap_dict = explainer(X_test)
                shap_values = explainer.shap_values(X_test)
                X_names = X_rf.rename(columns=var_names_dict)
                shap_values_df = pd.DataFrame(shap_values, columns=X_names.columns.tolist())
                save_csv_path = analysis_path + "Results/Predict_ahead_x_years/Importance/csv/Test/{}/".format(model)
                shap_values_df.to_csv(save_csv_path + "df{}_SHAP_values.csv".format(df_num))
                rf_shap_dfs_dict[df_num] = shap_values_df

                # plot
                n_features = 10

                plot_type = "bar"
                plot_SHAP(shap_dict, col_list=X_names.columns.tolist(),
                          n_features=n_features, plot_type=plot_type,
                          save_path= save_plot_path + "{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                plot_type = "summary"
                plot_SHAP(shap_dict, col_list=X_names.columns.tolist(),
                          n_features=n_features, plot_type=None,
                          save_path= save_plot_path + "{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                plot_type = "violin"
                plot_SHAP(shap_dict, col_list=X_names.columns.tolist(),
                          n_features=n_features, plot_type=plot_type,
                          save_path= save_plot_path + "{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))


if only_over_time == True:
    models = ["Enet", "RF"]
    for model in models:
        imp_data_path = analysis_path + "Results/Predict_ahead_x_years/Importance/csv/Test/{}/".format(model)
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

save_path = analysis_path + "Results/Predict_ahead_x_years/Importance/Plots/Over_time/"


if anal_level == "indiv":
    # plot average Enet
    if version == "_with_sges_T1":
        cut_off = 2300
    if version == "_without_sges_T1":
        cut_off = 4200
    enet_subgroup = shap_dict_to_long_df(dict=enet_shap_dfs_dict, cut_off = cut_off,
                                         csv_save_path=csv_save_path,
                                         csv_save_name = "SHAP_Enet_agg_time_cutoff{}_{}.csv".format(cut_off, start_string))

    save_path = analysis_path + "Results/Predict_ahead_x_years/Importance/Plots/Over_time/"

    if Include_T5 == False:
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
    if version == "_with_sges_T1":
        cut_off = 1500
    if version == "_without_sges_T1":
        cut_off = 1700
    rf_subgroup = shap_dict_to_long_df(dict=rf_shap_dfs_dict, cut_off=cut_off,
                                       csv_save_path=csv_save_path,
                                       csv_save_name= "SHAP_RF_agg_time.csv")

    if Include_T5 == False:
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
        if version == "_with_sges_T1":
            cut_off = 1
        if version == "_without_sges_T1":
            cut_off = 1
        enet_subgroup = shap_dict_to_long_df(dict=enet_shap_dfs_dict, cut_off=cut_off,
                                             csv_save_path=csv_save_path,
                                             csv_save_name="SHAP_Enet_agg_time_cutoff{}_{}.csv".format(cut_off, start_string))
        rf_subgroup = shap_dict_to_long_df(dict=rf_shap_dfs_dict, cut_off=cut_off,
                                           csv_save_path=csv_save_path,
                                           csv_save_name="SHAP_RF_agg_time_cutoff{}_{}.csv".format(cut_off, start_string))

        gamma = 0.3

        if Include_T5 == False:
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
    csv_block_save = analysis_path + "Results/Predict_ahead_x_years/Importance/csv/Test/Enet/block/"
    enet_group.to_csv(csv_block_save + "sum_importance_over_time.csv")

    if Include_T5 == False:
        enet_group.reset_index(inplace=True, drop=True)
        enet_group.drop(enet_group.loc[enet_group["Time"] == "5"].index,
                           inplace=True)
        t5 = "_without_t5"
    else:
        t5 = ""

    plot_SHAP_over_time(x="Time", y="Abs_SHAP", data=enet_group,
                        colour="block", save_path = save_path,
                        save_name = "Enet_block_abs_sum{}{}_{}".format(t5, changes, start_string),
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
    csv_block_save = analysis_path + "Results/Predict_ahead_x_years/Importance/csv/Test/RF/block/"
    rf_group.to_csv(csv_block_save + "sum_importance_over_time.csv")

    if Include_T5 == False:
        rf_group.reset_index(inplace=True, drop=True)
        rf_group.drop(rf_group.loc[rf_group["Time"] == "5"].index,
                         inplace=True)
        t5 = "_without_t5"
    else:
        t5 = ""

    plot_SHAP_over_time(x="Time", y="Abs_SHAP", data=rf_group,
                        colour="block", save_path = save_path,
                        save_name = "RF_block_abs_sum{}{}_{}".format(t5, changes, start_string),
                        xticks = xticks,
                        xlab="Schooling Year (Grade)",
                        ylab="SHAP Importance on Test Data",
                        title = "Random Forest", size=0.01,
                        fontsize=12,
                        palette=palette)

    # ===========================================================================
    # Plot relative to sges_T1 (rescale using importance of Maths ability Time 1):

    if Choose_drop_DV_T1 == False:
        models_norm_dict = {}
        df_dict = {'Enet': enet_group, 'RF': rf_group}
        for key, df in df_dict.items():
            normalised_dict = {}
            for time in times:
                time_chunk = df[df["Time"] == str(time)].reset_index().reset_index(drop=True)
                time_1_imp = time_chunk["Abs_SHAP"][time_chunk["block"] == "Maths Ability Time 1"].values[0]
                time_chunk.drop(time_chunk.loc[time_chunk['block'] == "Maths Ability Time 1"].index,
                                inplace=True)
                time_chunk["Normalised_Importance"] = round(time_chunk["Abs_SHAP"] / time_1_imp, 4)
                normalised_dict[time] = time_chunk
            normalised_df = pd.concat(normalised_dict, axis=0)
            if Include_T5 == False:
                normalised_df.drop(normalised_df.loc[normalised_df["Time"] == "5"].index, inplace=True)
                t5 = "_without_t5"
            else:
                t5 = ""
            models_norm_dict[key] = normalised_df

        # Enet
        var_names = pd.read_csv("Data/MetaData/variable_names.csv")
        palette = get_custom_palette(df=var_names, var_name="long_name")

        plot_long(x='Time', y='Normalised_Importance', data=models_norm_dict["Enet"], colour="block",
                  save_path=save_path, legend_loc="upper right",
                  save_name="Normed_SHAP_Importance_Enet_test_set_{}_t{}{}{}".format(time_date_string, cut_off, t5, changes),
                  xlab="School year (grade)", palette=palette,
                  ylab="SHAP Importance on Test Data (normalised)", xticks=xticks,
                  title="Elastic Net")
        # RF
        plot_long(x='Time', y='Normalised_Importance', data=models_norm_dict["RF"], colour="block",
                  save_path=save_path, legend_loc="upper right",
                  save_name="Normed_SHAP_Importance_RF_test_set_{}_t{}{}{}".format(time_date_string, cut_off, t5, changes),
                  xlab="School year (grade)", palette=palette,
                  ylab="SHAP Importance on Test Data (normalised)", xticks=xticks,
                  title="Random Forest")

    # heatmaps
    if heat_map == True:
        gamma = 0.3

        if Include_T5 == False:
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