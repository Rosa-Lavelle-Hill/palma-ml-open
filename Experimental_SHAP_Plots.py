
import pandas as pd
from Fixed_params import block_file

from Functions.Block_Permutation_Importance import parse_data_to_dict
from Functions.Plotting import plot_shap_scatt, plot_shap_scatt_looped, plot_shap_swarm_looped, \
    plot_shap_sns_swarm_looped
from Fixed_params import drop_list, parental_degree_expectations, drop_p_degree_expect, drop_houspan, cat_vars

invert_grades = True
only_all_together = True
Test = False
a = 1
if a == 1:
    file_path = "Analysis_3_with_sges_T1"
    start_string = "_25_Jan_2023__12.15"
if a == 2:
    file_path = "Predict_X_years_ahead_with_sges_T1"
    start_string = ""

model = "RF"

if drop_p_degree_expect == True:
    drop_list = drop_list + parental_degree_expectations
if drop_houspan == True:
    drop_list = drop_list + ["houspan"]
    if "houspan" in cat_vars:
        cat_vars.remove("houspan")

if Test == False:
    nrows=None
    t = ""
    dfs = ['1', '2', '3', '4']
if Test == True:
    nrows=10
    t = "_test"
    dfs = ['1', '2', '3', '4']

group_labels = ["Prior Ach.", "Grades", "Track", "Demo. & SES",
                "IQ", "Motiv. & Emot.", "Fam. Context (S,P)",
                "Cog. Strat.", "Class Context (S)", "Class Context (T)"]

if only_all_together == False:
    # loop through all dfs
    list_dfs = []
    long_df_dict = {}
    for df_num in dfs:
        print("df {}".format(df_num))
        df = pd.read_csv(file_path + "/Results/Importance/csv/Test_Data/{}/df{}_SHAP_values{}.csv".format(model, df_num,
                                                                                                        start_string),
                   index_col=[0], nrows=nrows)
        X_test = pd.read_csv(file_path + "/Modelling_Data/{}/df{}_X_test.csv".format(model, df_num),
                             index_col=[0], nrows=nrows)
        X_test_tr = pd.read_csv(file_path + "/Modelling_Data/{}/Transformed_preprocessor/".format(model) +\
                    "df{}_X_test_transformed.csv".format(df_num), index_col=[0], nrows=nrows)

        # recode binary variables on transform data 0s to -1:
        # Dictionary of Column name with associated index.
        idx_dic = {}
        for col in df.columns:
            idx_dic[col] = df.columns.get_loc(col)
        cat_dum_cols = list(X_test_tr.columns[76:])

        # recode dummy vars 0s to -1s
        X_test_tr[cat_dum_cols].replace(to_replace=0, value =-1, inplace=True)

        # remove ".0" from col names:
        new_names = []
        for col in X_test_tr.columns:
            new_col_name = col.replace(".0", "")
            new_names.append(new_col_name)
        X_test_tr.columns = new_names

        new_names = []
        for col in df.columns:
            new_col_name = col.replace(".0", "")
            new_names.append(new_col_name)
        df.columns = new_names

        block_dict = parse_data_to_dict(block_file)
        col_order_list = []
        new_block_dict = {}
        for block_name, block_vars in block_dict.items():
            new_block_vars = []
            for var in block_vars:
                if var in list(X_test_tr.columns):
                    new_block_vars.append(var)
                    col_order_list.append(var)
            new_block_dict[block_name] = new_block_vars

        # order:
        print(df.shape)
        df = df[col_order_list]
        print(df.shape)
        # todo: what feature gets dropped here and why?

        # long
        df.reset_index(inplace=True)
        df_long = pd.melt(df, id_vars="index")
        df_long["cat"] = '0'
        df_long["cat"][df_long["value"]<0] = "-1"
        df_long["cat"][df_long["value"]>0] = "1"

        # order:
        X_test_tr = X_test_tr[col_order_list]

        # long
        X_test_tr.reset_index(inplace=True)
        X_test_tr_long = pd.melt(X_test_tr, id_vars="index")

        # join
        df_long_all = df_long.merge(X_test_tr_long, on=["variable","index"])
        df_long_all.columns = ['index', 'variable', 'SHAP_value', "SHAP_cat", "data_value"]

        df_long_all["data_cat"] = '0'
        df_long_all["data_cat"][df_long_all["data_value"]<0] = "-1"
        df_long_all["data_cat"][df_long_all["data_value"]>0] = "1"

        # print(df_long_all["variable"].nunique())
        # df_long_all = df_long_all[df_long_all.SHAP_cat != '0']
        # print(df_long_all["variable"].nunique())

        # create 5 data categories:
        df_long_all["data_5_cat"] = '3'
        df_long_all["data_5_cat"].loc[df_long_all["data_value"]>1] = "1"
        df_long_all["data_5_cat"].loc[(df_long_all["data_value"] >= 0.5) & (df_long_all["data_value"] <= 1)] = "2"
        df_long_all["data_5_cat"].loc[(df_long_all["data_value"]<=-0.5) & (df_long_all["data_value"]>-1)] = "4"
        df_long_all["data_5_cat"].loc[df_long_all["data_value"]<=-1] = "5"

        lines = []
        line_int = -0.5
        for block, vars_list in new_block_dict.items():
            block_len = len(vars_list)
            line_int = line_int + block_len
            lines.append(line_int)

        pd.DataFrame(lines).to_csv(file_path + "/Results/Importance/csv/Test_Data/{}/directional_plotting/lines.csv".format(model))

        label_geo_list=[]
        line_int = 0
        for block, vars_list in new_block_dict.items():
            block_len = len(vars_list)
            # if block_len > 1:
            midpoint = block_len/2
            # elif block_len == 2:
            #     midpoint = 0.7
            # else:
            #     midpoint = 0.5
            line_geo = line_int + midpoint
            label_geo_list.append(line_geo)
            line_int = line_int + block_len

        pd.DataFrame(label_geo_list).to_csv(
            file_path + "/Results/Importance/csv/Test_Data/{}/directional_plotting/label_geo_list.csv".format(model))

        save_path = file_path + "/Results/Importance/Plots/Test_Data/{}/directional/".format(model)
        plot_shap_scatt(data=df_long_all, x="variable", y="SHAP_value",
                        hue="data_5_cat",
                        save_path=save_path,
                        save_name="{}_df{}{}{}".format(model, df_num, start_string, t),
                        xlab="Predictor variables", ylab="SHAP value",
                        lines=lines, label_geo=label_geo_list,
                        group_labels=group_labels
                        )
        df_long_all['Time'] = df_num
        df_long_all.to_csv(file_path + "/Results/Importance/csv/Test_Data/{}/directional_plotting/long_df_{}{}.csv".format(model, df_num, t))
        # add people variance
        long_df_dict[df_num] = df_long_all
        list_dfs.append(df_long_all)

    df_everything = pd.concat(list_dfs, axis=0, ignore_index=True)
    df_everything.to_csv(file_path + "/Results/Importance/csv/Test_Data/{}/directional_plotting/long_all_dfs{}.csv".format(model, t))

if only_all_together == True:
    df_everything = pd.read_csv(file_path + "/Results/Importance/csv/Test_Data/{}/directional_plotting/long_all_dfs{}.csv".format(model, t), index_col=[0])
    lines = pd.read_csv(file_path + "/Results/Importance/csv/Test_Data/{}/directional_plotting/lines.csv".format(model),
                        index_col=[0])
    lines = list(lines["0"])
    label_geo_list = pd.read_csv(file_path + "/Results/Importance/csv/Test_Data/{}/directional_plotting/label_geo_list.csv".format(model),
                        index_col=[0])
    label_geo_list = list(label_geo_list["0"])

save_path = file_path + "/Results/Importance/Plots/Test_Data/{}/directional/".format(model)

if invert_grades == True:
    df_old = df_everything.copy()
    grades = ["gsp_jz", "gmu_jz", "gbi_jz", "gma_jz", "gde_jz", "gla_jz"]
    for grade in grades:
        old_shap_value = df_everything[df_everything['variable'] == grade]['SHAP_value']
        new_shap_value = old_shap_value * -1
        old_grade_value = df_everything[df_everything['variable'] == grade]['data_value']
        new_grade_value = old_grade_value * -1
        df_everything.loc[df_everything['variable'] == grade, 'SHAP_value'] = df_everything.loc[df_everything['variable'] == grade, 'SHAP_value'] * -1
        df_everything.loc[df_everything['variable'] == grade, 'data_value'] = df_everything.loc[df_everything['variable'] == grade, 'data_value'] * -1

plot_shap_scatt_looped(df_everything, x="variable", y="SHAP_value",
                    hue="data_value",
                    save_path=save_path,
                    save_name="REV_{}_dfs_all_together{}{}".format(model, start_string, t),
                    xlab="Grouped predictor variables", ylab="SHAP value",
                    lines=lines, label_geo=label_geo_list,
                    group_labels=group_labels
                    )

plot_shap_sns_swarm_looped(df_everything, x="variable",
                           y="SHAP_value",
                            hue="data_5_cat",
                            save_path=save_path,
                            save_name="REV_vio_{}_dfs_all_together{}{}".format(model, start_string, t),
                            xlab="Grouped predictor variables", ylab="SHAP value",
                            lines=lines, label_geo=label_geo_list,
                            group_labels=group_labels
                    )

print('done!')