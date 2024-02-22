from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from Fixed_params import imputer_model, imputer_max_iter, drop_list, parental_degree_expectations, cat_vars,\
    drop_p_degree_expect, drop_houspan
from Functions.Generic_Fit_Transform import get_preprocessed_col_names
from Functions.Plotting import plot_SHAP_interaction, plot_SHAP_summary_interaction, plot_SHAP_interact_dependency
from Functions.Predict import build_pipeline_enet, build_pipeline_rf
from Functions.RF_preprocess import to_categorical


def SHAP_enet(X, y, df_num, save_csv_path, enet_shap_dfs_dict,
              enet_params, modelling_data_save, drop_list=drop_list,
              start_string = ""):
    model = "Enet"
    print("running SHAP {} function for df{}".format(model, df_num))
    if drop_p_degree_expect == True:
        drop_list = drop_list + parental_degree_expectations
    if drop_houspan == True:
        drop_list = drop_list + ["houspan"]
        if "houspan" in cat_vars:
            cat_vars.remove("houspan")

    # code categorical vars:
    X = to_categorical(X, cat_vars)

    # if missing, fill houspan (parents) with housechan (child):
    if drop_houspan == False:
        X["houspan"].fillna(X["houschn"], inplace=True)

    for col in X.columns:
        if col in drop_list:
            X.drop(col, axis=1, inplace=True)

    pipe, preprocessor = build_pipeline_enet(X=X, imputer_model=imputer_model,
                                             oh_encoder_drop='first',
                                             imputer_max_iter=imputer_max_iter)

    pipe.set_params(**enet_params)

    # split data using same params (should be same as the train/test set used for prediction)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)
    print("y train n={}; mean={},\ny test n={}; mean={}".format(y_train.shape[0], round(y_train.mean(), 2),
                                                                y_test.shape[0], round(y_test.mean(), 2)))
    # save data sets:
    X_train.to_csv(modelling_data_save + "{}/df{}_X_train.csv".format(model, df_num))
    X_test.to_csv(modelling_data_save + "{}/df{}_X_test.csv".format(model, df_num))
    pd.DataFrame(y_train).to_csv(modelling_data_save + "{}/df{}_y_train.csv".format(model, df_num))
    pd.DataFrame(y_test).to_csv(modelling_data_save + "{}/df{}_y_test.csv".format(model, df_num))

    elastic_net_regression = ElasticNet(alpha=enet_params['regression__alpha'],
                                        l1_ratio=enet_params['regression__l1_ratio'],
                                        max_iter=enet_params['regression__max_iter'],
                                        tol=enet_params['regression__tol'])
    print('Fitting Elastic Net...')
    # fit to and transform train
    X_train = preprocessor.fit_transform(X_train)
    elastic_net_regression.fit(X_train, y_train)
    # transform test
    X_test = preprocessor.transform(X_test)
    # get processed data col names:
    names = get_preprocessed_col_names(X, pipe)
    X_test_tr_df = pd.DataFrame(X_test, columns=names)
    X_test_tr_df.to_csv(modelling_data_save + "/{}/Transformed_preprocessor/".format(model) +
                  "df{}_X_test_transformed.csv".format(df_num))
    # Fit the explainer
    explainer = shap.LinearExplainer(elastic_net_regression, X_test)

    # Calculate the SHAP values and save
    shap_dict = explainer(X_test)
    shap_values = explainer.shap_values(X_test)
    shap_values_df = pd.DataFrame(shap_values, columns=names)
    shap_values_df.to_csv(save_csv_path + "{}/".format(model) + "df{}_SHAP_values{}.csv".format(df_num, start_string))
    enet_shap_dfs_dict[df_num] = shap_values_df
    return shap_dict, names


def SHAP_rf(X, y, df_num, save_csv_path, rf_shap_dfs_dict,
              rf_params, modelling_data_save, drop_list=drop_list,
              drop_p_degree_expect=True, drop_houspan=drop_houspan, start_string = "",
              ):
    model = "RF"
    print("running SHAP {} function for df{}".format(model, df_num))
    if drop_p_degree_expect == True:
        drop_list = drop_list + parental_degree_expectations
    if drop_houspan == True:
        drop_list = drop_list + ["houspan"]
        if "houspan" in cat_vars:
            cat_vars.remove("houspan")

    # code categorical vars:
    X = to_categorical(X, cat_vars)

    # if missing, fill houspan (parents) with housechan (child):
    if drop_houspan == False:
        X.loc[X['houspan'].isnull(), 'houspan'] = X['houschn']

    for col in X.columns:
        if col in drop_list:
            X.drop(col, axis=1, inplace=True)

    pipe, preprocessor = build_pipeline_rf(X=X, imputer_model=imputer_model,
                                             oh_encoder_drop='if_binary',
                                             imputer_max_iter=imputer_max_iter)

    pipe.set_params(**rf_params)

    # split data using same params (should be same as the train/test set used for prediction)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2, shuffle=True)
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
                                                     n_estimators=rf_params['regression__n_estimators'])
    print('Fitting Random Forest...')
    # fit to and transform train
    X_train = preprocessor.fit_transform(X_train)
    random_forest_regression.fit(X_train, y_train)
    # transform test
    X_test = preprocessor.transform(X_test)
    # get processed data col names:
    names = get_preprocessed_col_names(X, pipe)
    X_test_tr_df = pd.DataFrame(X_test, columns=names)
    X_test_tr_df.to_csv(modelling_data_save + "{}/Transformed_preprocessor/".format(model) +
                  "df{}_X_test_transformed.csv".format(df_num))
    # Fit the explainer
    explainer = shap.TreeExplainer(random_forest_regression, X_test)


    # Calculate the SHAP values and save
    shap_dict = explainer(X_test)
    shap_values = explainer.shap_values(X_test)
    shap_values_df = pd.DataFrame(shap_values, columns=names)
    shap_values_df.to_csv(save_csv_path + "{}/".format(model) + "df{}_SHAP_values{}.csv".format(df_num, start_string))
    rf_shap_dfs_dict[df_num] = shap_values_df
    return shap_dict, names



def SHAP_tree_interaction(save_path, X, y, df_num, start_string,
              rf_params, names, n_inter_features=10,
                          vars1=["School Track", "IQ"],
                          vars2=["Maths Ability Time 1", "Maths Ability Time 1"]):
    model = "RF"
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=93, test_size=0.2,
                                                        shuffle=True)
    random_forest_regression = RandomForestRegressor(max_depth=rf_params['regression__max_depth'],
                                                     max_features=rf_params['regression__max_features'],
                                                     min_samples_split=rf_params[
                                                         'regression__min_samples_split'],
                                                     random_state=rf_params['regression__random_state'],
                                                     n_estimators=rf_params['regression__n_estimators'])
    explainer = shap.TreeExplainer(random_forest_regression, X_test)
    shap_values = explainer.shap_values(X_test)
    shap_values_df = pd.DataFrame(shap_values, columns=names)

    shap_interaction = explainer.shap_interaction_values(X_test)
    plot_cols = pd.DataFrame(abs(shap_values_df).sum(axis=0)).sort_values(ascending=False, by=0)[
                0:n_inter_features].index.values
    plot_cols_list = list(plot_cols)
    if 'Maths Ability Time 1' not in plot_cols_list:
        plot_cols_list = plot_cols_list + ['Maths Ability Time 1']
    plot_SHAP_interaction(shap_interaction, save_path=save_path,
                          col_list=names, fontsize=8, plot_cols=plot_cols_list,
                          save_name="SHAP_interaction_df{}_{}_{}_n{}".format(df_num,
                                                                             model,
                                                                             start_string,
                                                                             n_inter_features))
    plot_SHAP_summary_interaction(shap_interaction, X=X_test, save_path=save_path,
                                  col_list=names, fontsize=8,
                                  save_name="SHAP_summary_interaction_df{}_{}_{}_n{}".format(df_num,
                                                                                             model,
                                                                                             start_string,
                                                                                             n_inter_features))
    for var1, var2 in zip(vars1, vars2):
        X_test_df = pd.DataFrame(X_test, columns=names)
        plot_SHAP_interact_dependency(shap_interaction, X=X_test_df, save_path=save_path,
                                      var1=var1,
                                      var2=var2, fontsize=8,
                                      save_name="SHAP_depend_{}_{}_df{}_{}_{}_n{}".format(var1,
                                                                                          var2,
                                                                                          df_num,
                                                                                          model,
                                                                                          start_string,
                                                                                          n_inter_features))
