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
from Functions.Plotting import plot_perm_box, plot_perm, plot_SHAP, plot_SHAP_over_time, get_custom_palette
from Functions.Preprocessing import get_var_names_dict, remove_col, shap_dict_to_long_df, get_var_names_dict_agg
from RUN_Analysis_7 import start_string, global_seed, analysis_path, preoptimised_model, version, anal_level,\
    data_path, directory_bits, enet_cut_off, rf_cut_off, use_preoptimised_model
from Fixed_params import cat_vars, drop_list

shap.initjs()

start_time = dt.datetime.now()
# ----
only_over_time = False
data_used = 'test'
n_plot=25

np.random.seed(global_seed)
time_date_string = start_string

# -------
if use_preoptimised_model == True:
    load_analysis_path = "Analysis_{}{}/".format(preoptimised_model, version)
    rf_param_path = load_analysis_path + "Results/Best_params/RF/"
    enet_param_path = load_analysis_path + "Results/Best_params/Enet/"

# get custom colour palette:
var_names = pd.read_csv("Data/MetaData/variable_names.csv")
var_names_dict = get_var_names_dict_agg(var_names)
var_names_agg = pd.DataFrame.from_dict(var_names_dict, orient='index').reset_index()
var_names_agg.columns = ["short_name", "long_name"]
palette = get_custom_palette(df=var_names_agg, var_name="long_name")

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

modelling_data_save = analysis_path + "Modelling_Data/"

rf_shap_dfs_dict = {}
enet_shap_dfs_dict = {}
if only_over_time == False:

    # loop through dfs:
    for file in os.listdir(directory_bits):
        filename = os.fsdecode(file)
        if filename.endswith("{}.csv".format(version)):
            start_df = dt.datetime.now()
            print("--> Processing.... {}".format(filename))
            df_num = [s for s in re.findall(r'\d+', filename)][0]

            df = pd.read_csv(data_path+filename)

            # code categorical vars:
            df = to_categorical(df, cat_vars)

            for col in df.columns:
                if col in drop_list:
                    df.drop(col, axis=1, inplace=True)

            # pipeline steps:
            imputer = IterativeImputer(missing_values=np.nan, max_iter=10)
            transformer = StandardScaler()

            # additional preprocess steps for Enet:
            df_enet = dummy_code(df=df, vars_to_code=cat_vars)

            # define X and y:
            y_enet = df_enet["sges"]
            X_enet = df_enet.drop("sges", axis=1)
            X_enet = remove_special_char_colnames(X_enet, '.')

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

                # fit each step of the pipeline on train data, transform test data

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
                save_csv_path = analysis_path + "Results/Importance/csv/Test_Data/{}/".format(model)
                shap_values_df.to_csv(save_csv_path + "df{}_SHAP_values.csv".format(df_num))
                enet_shap_dfs_dict[df_num] = shap_values_df

                # plot
                n_features = 10

                plot_type = "bar"
                plot_SHAP(shap_dict, col_list=X_names.columns.tolist(),
                          n_features=n_features, plot_type=plot_type,
                          save_path= analysis_path + "Results/Importance/Plots/Test_Data/{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                plot_type = "summary"
                plot_SHAP(shap_dict, col_list=X_names.columns.tolist(),
                          n_features=n_features, plot_type=None,
                          save_path= analysis_path + "Results/Importance/Plots/Test_Data/{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                plot_type = "violin"
                plot_SHAP(shap_dict, col_list=X_names.columns.tolist(),
                          n_features=n_features, plot_type=plot_type,
                          save_path= analysis_path + "Results/Importance/Plots/Test_Data/{}/{}/".format(model, plot_type),
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

                print('Fitting Random Forest...')

                random_forest_regression = RandomForestRegressor(max_depth=rf_params['regression__max_depth'],
                                                    max_features=rf_params['regression__max_features'],
                                                    min_samples_split=rf_params['regression__min_samples_split'],
                                                    random_state=rf_params['regression__random_state'],
                                                    n_estimators= rf_params['regression__n_estimators'])

                # fit to and transform train
                X_train = imputer.fit_transform(X_train)
                X_train = transformer.fit_transform(X_train)
                random_forest_regression.fit(X_train, y_train)

                # transform test
                X_test = imputer.transform(X_test)
                X_test = transformer.transform(X_test)

                # Fit the explainer
                explainer = shap.TreeExplainer(random_forest_regression, X_test, feature_pertubation="tree_path_dependent")

                # Calculate the SHAP values and save
                shap_dict = explainer(X_test, check_additivity=False)
                shap_values = explainer.shap_values(X_test, check_additivity=False)
                X_names = X_rf.rename(columns=var_names_dict)
                shap_values_df = pd.DataFrame(shap_values, columns=X_names.columns.tolist())
                save_csv_path = analysis_path + "Results/Importance/csv/Test_Data/{}/".format(model)
                shap_values_df.to_csv(save_csv_path + "df{}_SHAP_values.csv".format(df_num))
                rf_shap_dfs_dict[df_num] = shap_values_df

                # plot
                n_features = 10

                plot_type = "bar"
                plot_SHAP(shap_dict, col_list=X_names.columns.tolist(),
                          n_features=n_features, plot_type=plot_type,
                          save_path= analysis_path + "Results/Importance/Plots/Test_Data/{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                plot_type = "summary"
                plot_SHAP(shap_dict, col_list=X_names.columns.tolist(),
                          n_features=n_features, plot_type=None,
                          save_path= analysis_path + "Results/Importance/Plots/Test_Data/{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

                plot_type = "violin"
                plot_SHAP(shap_dict, col_list=X_names.columns.tolist(),
                          n_features=n_features, plot_type=plot_type,
                          save_path= analysis_path + "Results/Importance/Plots/Test_Data/{}/{}/".format(model, plot_type),
                          save_name="df{}_{}_SHAP_nfeatures{}".format(df_num, model, n_features))

if only_over_time == True:
    models = ["Enet", "RF"]
    for model in models:
        data_path = analysis_path + "Results/Importance/csv/Test_Data/{}/".format(model)
        directory_bits = os.fsencode(data_path)
        for file in os.listdir(directory_bits):
            filename = os.fsdecode(file)
            if filename.endswith("SHAP_values.csv"):
                df_num = [s for s in re.findall(r'\d+', filename)][0]
                df = pd.read_csv(data_path + filename, index_col=[0])
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
xticks = ["5 --> 6", "6 --> 7", "7 -- >8", "8 --> 9", "9 --> 10"]

if anal_level == "indiv":
    # plot average Enet
    enet_subgroup = shap_dict_to_long_df(dict=enet_shap_dfs_dict, cut_off = enet_cut_off,
                                         csv_save_path=csv_save_path_test,
                                         csv_save_name = "SHAP_Enet_agg_time.csv")

    save_path = analysis_path + "Results/Importance/Plots/Over_time/"
    xticks = ["5 --> 6", "6 --> 7", "7 -- >8", "8 --> 9", "9 --> 10"]

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
    rf_subgroup = shap_dict_to_long_df(dict=rf_shap_dfs_dict, cut_off=rf_cut_off,
                                       csv_save_path=csv_save_path_test,
                                       csv_save_name= "SHAP_RF_agg_time.csv")

    plot_SHAP_over_time(x="Time", y="Abs_SHAP", data=rf_subgroup,
                        colour="variable", save_path = save_path,
                        save_name = "RF_abs_sum",
                        xticks = xticks,
                        xlab="Schooling Year (Grade)",
                        ylab="SHAP Importance on Test Data",
                        title = "", size=0.01,
                        fontsize=12, palette=palette)


# todo: then plot variance, and pos/neg for each variable
# =========================
# At group-level
# =========================
if anal_level == "block":
    # load variable groups
    if add_teacher_vars == True:
        block_file = "Data/MetaData/Variable_blocks/variable_blocks_final_with_Teacher.csv"
    if add_teacher_vars == False:
        block_file = "Data/MetaData/Variable_blocks/variable_blocks_final.csv"

    block_dict = parse_data_to_dict(block_file)

    # Enet
    enet_subgroup_agg = shap_dict_to_long_df(dict=enet_shap_dfs_dict, cut_off = 0,
                                         csv_save_path=csv_save_path_test,
                                         csv_save_name = "SHAP_Enet_agg_time.csv")

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

    plot_SHAP_over_time(x="Time", y="Abs_SHAP", data=enet_group,
                        colour="block", save_path = save_path,
                        save_name = "Enet_block_abs_sum",
                        xticks = xticks,
                        xlab="Schooling Year (Grade)",
                        ylab="SHAP Importance on Test Data",
                        title = "", size=0.01,
                        fontsize=12,
                        palette=palette)
    # RF
    rf_subgroup_agg = shap_dict_to_long_df(dict=rf_shap_dfs_dict, cut_off = 0,
                                         csv_save_path=csv_save_path_test,
                                         csv_save_name = "SHAP_RF_agg_time.csv")

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

    plot_SHAP_over_time(x="Time", y="Abs_SHAP", data=rf_group,
                        colour="block", save_path = save_path,
                        save_name = "RF_block_abs_sum",
                        xticks = xticks,
                        xlab="Schooling Year (Grade)",
                        ylab="SHAP Importance on Test Data",
                        title = "", size=0.01,
                        fontsize=12,
                        palette=palette)

# =========================
end_time = dt.datetime.now()
run_time = end_time - start_time
print("SHAP importance runtime: {}".format(run_time))

print('done!')