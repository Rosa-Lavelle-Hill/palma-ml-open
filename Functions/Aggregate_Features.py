import re
import random
import os
import numpy as np
import pandas as pd
from Functions.Preprocessing import remove_cols
from Fixed_params import agg_mode, cat_vars, teacher_cols

center_function = lambda x: x - x.mean()

def Class_level(directory_bits, data_path, aggregate_meta_data, save_path, version,
                center_function=center_function, save_name = "aggregate_preprocessed"):
    aggregate_meta_data['School_Class'] = pd.Series(aggregate_meta_data["School_Code"].astype(str) + "_"
                                                    + aggregate_meta_data["Class_Code"].astype(str))
    for file in os.listdir(directory_bits):
        filename = os.fsdecode(file)
        if filename.endswith("_preprocessed{}.csv".format(version)):
            print("--> Processing.... {}".format(filename))
            df_num = [s for s in re.findall(r'\d+', filename)][0]

            df = pd.read_csv(data_path + filename, index_col=[0])

            # rejoin meta aggreagte data
            wdf = df.merge(aggregate_meta_data[["vpnr", "School_Class"]], how='left', on="vpnr")

            # Next calculate school_class (class) aggregate level:
            wdf_class_agg = wdf.copy()
            wdf_class_agg.drop('vpnr', inplace=True, axis=1)

            # mean only for vars not in "agg_mode"
            wdf_class_agg_cont = wdf_class_agg.loc[:, ~wdf_class_agg.columns.isin(agg_mode)].\
                groupby(['School_Class']).mean().reset_index()

            # for cat vars, use mode:
            include = ['School_Class'] + agg_mode
            wdf_class_agg_cat = wdf_class_agg.loc[:, wdf_class_agg.columns.isin(include)].\
                groupby(
                ['School_Class']).agg(pd.Series.mode).reset_index()

            # if multiple modes, randomly select one:
            for index, row in wdf_class_agg_cat.loc[:, wdf_class_agg_cat.columns != "School_Class"].iterrows():
                d = row.to_dict()
                for i, k in row.to_dict().items():
                    if isinstance(k, np.ndarray):
                        if len(k) > 1:
                            d[i] = random.sample(list(k), 1)[0]
                        else:
                            d[i] = np.nan
                    else:
                        d[i] = k
                new_row = pd.Series(d)
                wdf_class_agg_cat.loc[index, wdf_class_agg_cat.columns != "School_Class"] = new_row

            # join agg mean and agg mode dfs together:
            wdf_class_agg = wdf_class_agg_cont.merge(wdf_class_agg_cat, on="School_Class")

            # get new col names
            orig_name_list = list(wdf_class_agg.columns.values)[1:]
            class_col_name_list = []
            for col in wdf_class_agg.columns[1:]:
                col_name = col + '_'
                class_col_name = col_name + "C"
                class_col_name_list.append(class_col_name)

            # rename agg cols:
            rename_class_dict = dict(zip(orig_name_list, class_col_name_list))
            wdf_class_agg.rename(rename_class_dict, inplace=True, axis=1)

            # mean center indiv level features:
            c_cols = wdf.columns.tolist()
            # items to be removed
            unwanted = {'vpnr', 'Unnamed: 0.1', 'School_Class'}
            c_cols = [ele for ele in c_cols if ele not in unwanted]

            # don't mean center cat vars:
            cat_vars_remove = set(cat_vars)
            c_cols = [ele for ele in c_cols if ele not in cat_vars_remove]
            centered = center_function(wdf[c_cols])
            non_centered_cols = ['vpnr', 'School_Class'] + cat_vars
            centered_df = wdf[non_centered_cols].join(centered)

            # merge datasets
            agg_df = centered_df.merge(wdf_class_agg, on="School_Class")
            removes = ["sges_C", "vpnr_C", "School_Class", "vpnr"]
            agg_df = remove_cols(agg_df, removes)
            print("df{}: {}".format(df_num, agg_df.shape))

            # remove agg teacher cols, as the aggregate should be same as original cols (but more accurate)
            teacher_col_name_list = []
            for i in list(teacher_cols['0']):
                col_name = i + '_'
                teacher_col_name = col_name + "C"
                teacher_col_name_list.append(teacher_col_name)
            agg_df = remove_cols(agg_df, teacher_col_name_list)

            # save cols list:
            cols_list_save = "Data/Initial_Preprocess/Cols_lists/"
            col_list = pd.DataFrame(agg_df.columns)
            col_list.to_csv(cols_list_save + "df{}_aggregate_cols.csv".format(df_num))

            # save new agg data
            agg_df.to_csv(save_path + "df{}_".format(df_num) + save_name + version + ".csv")

    return



