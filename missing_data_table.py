import pandas as pd
from Fixed_params import block_file
from Functions.Block_Permutation_Importance import parse_data_to_dict

round_n = 2
# A1
data_path = "Analysis_2_with_sges_T1/Processed_Data/"
data_groups = pd.read_csv("Data/MetaData/Variable_blocks/variable_blocks_final_with_Teacher_change_cog_and_track.csv")
info_dict = {}
data_dict = {}
teacher_dict = {}
teacher_info_dict = {}
parent_dict = {}
parent_info_dict = {}

block_dict = parse_data_to_dict(block_file)
teacher_cols = block_dict['Teacher_reported_classroom_context']
teacher_cols.remove('sext_2.0')
teacher_cols.remove('aget')
parent_cols = ['factp', 'fvalp', 'ausup']

for df_num in ['1', '2', '3', '4']:
    df_dict = {}
    teacher_dict = {}
    df = pd.read_csv(data_path + "df{}_preprocessed_with_sges_T1.csv".format(df_num))
    data_dict[df_num] = df

    miss_perc = round(df.isnull().sum().sum() / len(df), round_n)
    sex_count = df["sex"].value_counts()
    sex_count_total = sex_count.sum()
    sex_count_df = pd.DataFrame(df["sex"].value_counts()).reset_index()
    perc_F = round((sex_count_df[sex_count_df['index']==1]['sex'].values[0] / sex_count_total) * 100, round_n)

    # look at some teacher vars
    teacher_sex_miss = df['sext'].isnull().sum()
    teacher_enth_miss = df['entht'].isnull().sum()
    teacher_movt_miss = df['movt'].isnull().sum()

    teacher_dict['teacher_est_miss'] = teacher_sex_miss
    teacher_dict['teacher_enth_miss'] = teacher_enth_miss
    teacher_dict['teacher_movt_miss'] = teacher_movt_miss

    # across all teacher vars:
    all_teacher_miss = df[teacher_cols].isnull().sum().sum()
    all_teacher = df[teacher_cols].shape[0] * df[teacher_cols].shape[1]
    teacher_perc = round((all_teacher_miss/all_teacher) * 100, round_n)

    # look at some parent vars
    par_factp_miss = df['factp'].isnull().sum()
    par_ausup_miss = df['ausup'].isnull().sum()
    par_fvalp_miss = df['fvalp'].isnull().sum()

    parent_dict['par_factp_miss'] = par_factp_miss
    parent_dict['par_ausup_miss'] = par_ausup_miss
    parent_dict['par_fvalp_miss'] = par_fvalp_miss

    # across all parent vars:
    all_parent_miss = df[parent_cols].isnull().sum().sum()
    all_parent = df[parent_cols].shape[0] * df[parent_cols].shape[1]
    parent_perc = round((all_parent_miss/all_parent) * 100, round_n)

    # summary stats:
    df_dict['N'] = df.shape[0]
    df_dict['F_perc'] = perc_F
    df_dict['teacher_miss_perc'] = teacher_perc
    df_dict['parent_miss_perc'] = parent_perc
    df_dict['miss_perc'] = miss_perc

    info_dict[df_num] = df_dict
    teacher_info_dict[df_num] = teacher_dict
    parent_info_dict[df_num] = parent_dict


sum_df = pd.DataFrame.from_dict(info_dict).T.reset_index()
sum_df.rename(columns={"index":"df"}, inplace=True)

# A2
data_path = "Predict_X_years_ahead_with_sges_T1/Processed_Data/Processed_Multicollinearity/"
info_dict = {}
for df_num in ['1', '2', '3', '4']:
    print(df_num)
    df_dict = {}
    df = pd.read_csv(data_path + "df{}_preprocessed_with_sges_T1.csv".format(df_num))
    miss_perc = round(df.isnull().sum().sum() / len(df), round_n)
    sex_count = df["sex"].value_counts()
    sex_count_total = sex_count.sum()
    sex_count_df = pd.DataFrame(df["sex"].value_counts()).reset_index()
    perc_F = round((sex_count_df[sex_count_df['index']==1]['sex'].values[0] / sex_count_total) * 100, round_n)

    # across all parent vars:
    all_parent_miss = df[parent_cols].isnull().sum().sum()
    print(all_parent_miss)
    all_parent = df[parent_cols].shape[0] * df[parent_cols].shape[1]
    print(all_parent)
    parent_perc = round((all_parent_miss/all_parent) * 100, round_n)
    print(parent_perc)

    # across all teacher vars:
    all_teacher_miss = df[teacher_cols].isnull().sum().sum()
    all_teacher = df[teacher_cols].shape[0] * df[teacher_cols].shape[1]
    # print(all_teacher)
    teacher_perc = round((all_teacher_miss/all_teacher) * 100, round_n)

    df_dict['N'] = df.shape[0]
    df_dict['F_perc'] = perc_F
    df_dict['teacher_miss_perc'] = teacher_perc
    df_dict['parent_miss_perc'] = parent_perc
    df_dict['miss_perc'] = miss_perc
    info_dict[df_num] = df_dict

sum_df2 = pd.DataFrame.from_dict(info_dict).T.reset_index()
sum_df2.rename(columns={"index":"df"}, inplace=True)

# join dfs and save as csv
df_merge = pd.concat([sum_df, sum_df2], axis=1)
df_merge.to_csv("Data/MetaData/descriptives_tab.csv")
print('done')