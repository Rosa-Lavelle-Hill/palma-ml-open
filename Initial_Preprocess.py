import re
import os
import pandas as pd
import numpy as np
import pyreadstat
import datetime as dt
import itertools
import scipy
import scipy.stats
import pingouin as pg
from statsmodels.formula.api import mixedlm


from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols

from Fixed_params import cat_vars, block_file
from Functions.Block_Permutation_Importance import parse_data_to_dict
from Functions.Plotting import plot_hist, plot_bar_h, plot_bar_h_df, plot_scatt
from Functions.Preprocessing import check_val, extract_constructs_df, \
    remove_reversed_teacher_vars, rename_teacher_vars, \
    remove_col, remove_cols, reaggregate_teacher_vars, find_cat_vars

# set DV:
dependent_variable = "sges"
dv_T1 = dependent_variable + '_T1'

# set params
missing_data_col_cutoff = 50
preprocess_drop_DV_T1 = False

if preprocess_drop_DV_T1 == True:
    version = "_without_{}_T1".format(dependent_variable)
if preprocess_drop_DV_T1 == False:
    version = "_with_{}_T1".format(dependent_variable)

# set jobs:
final_is_4_timepoints = True
school_track_change_var = True
add_immigration = True
recalc_aus = True
renaming = True
decode_pp_id = True
add_teacher_vars = True
do_extract_constructs = True
general_list = True
translate = False
plot_original_IVs = False
plot_IVs_by_wave = False
missing = True
plot_missing_data_by_wave = True
by_wave_plotting = False
# ^ takes 30sec, set to false for fast version
check_DV_descriptives = True
find_consistent_constructs = True
create_anal_df = True
plot_missing_5_datasets = True
final_steps = True

# --
if __name__ == "__main__":
    start = dt.datetime.now()

    file_name = "Data/All.sav"

    df_all, meta = pyreadstat.read_sav(file_name, encoding="latin1")

    # Create a dictionary of column_names:column_labels
    column_labels = meta.column_names_to_labels

    # find cat vars:
    cat_vars_dict = find_cat_vars(meta=meta)
    cat_vars_df = pd.DataFrame.from_dict(cat_vars_dict, orient='index')
    cat_vars_df.to_csv("Data/Initial_Preprocess/cat_vars_check.csv")

    # Integrate missing data from wave 2:
    file_name = "Data/MZP2_selected variables.sav"
    df_selected, meta_selected = pyreadstat.read_sav(file_name, encoding="latin1")
    df_all = pd.merge(df_all, df_selected, how='left', on='vpnr')

    # preprocess teacher data:
    if add_teacher_vars == True:
        teacher_path = "Data/PALMA_Lehrerdaten/"
        t1_orig, meta_t1 = pyreadstat.read_sav(teacher_path + "Lehrer_MZP1.sav", encoding="latin1")
        t2_orig, meta_t2 = pyreadstat.read_sav(teacher_path + "Lehrer_MZP2.sav", encoding="latin1")
        t3_orig, meta_t3 = pyreadstat.read_sav(teacher_path + "Lehrer_MZP3.sav", encoding="latin1")
        t4_orig, meta_t4 = pyreadstat.read_sav(teacher_path + "Lehrer_MZP4.sav", encoding="latin1")
        t5_orig, meta_t5 = pyreadstat.read_sav(teacher_path + "Lehrer_MZP5.sav", encoding="latin1")
        t6_orig, meta_t6 = pyreadstat.read_sav(teacher_path + "Lehrer_MZP6.sav", encoding="latin1")

        # fix some syntax issues with df3
        t3_orig.rename(columns={"idcode":"clnr"}, inplace=True)
        t3_orig['clnr'] = t3_orig['clnr'].astype('float')
        t3_orig.dropna(how='all', axis=1, inplace=True)

        # re-aggreagte t3 vars:
        t3_orig = reaggregate_teacher_vars(t3_orig)

        # preprocess teacher dfs
        teacher_wave_dict = {}
        teacher_dfs_orig = [t1_orig, t2_orig, t3_orig, t4_orig, t5_orig, t6_orig]
        for index, tdf in enumerate(teacher_dfs_orig):
            key = index + 1
            teacher_wave_dict[key] = tdf.copy()
            teacher_wave_dict[key] = extract_constructs_df(teacher_wave_dict[key])
            teacher_wave_dict[key] = remove_reversed_teacher_vars(teacher_wave_dict[key])
            teacher_wave_dict[key] = rename_teacher_vars(teacher_wave_dict[key])

        # count total teachers:
        unique_cases = set()

        # iterate over each data frame
        for k, df in teacher_wave_dict.items():
            id_col = df["clnr"]
            # add the column values to the set of unique cases
            unique_cases.update(id_col.tolist())

        # get the count of unique cases
        count = len(unique_cases)
        print("Unique teachers:", count)

        for index, tdf in enumerate(teacher_dfs_orig):
            key = index + 1
            teacher_wave_dict[key] = remove_cols(teacher_wave_dict[key], drop_cols=['klassnr', 'klassid', "vpnr"])

        # save teacher vars:
        teacher_save_path = "Data/MetaData/Constructs/Teacher/"
        teacher_cols_df = pd.DataFrame(dict([(k, pd.Series(v.columns)) for k, v in teacher_wave_dict.items()]))
        teacher_cols_df.to_csv(teacher_save_path + "Teacher_cols.csv")

        # find constructs across all 6:
        teacher_constructs_in_all_6_list = list(set.intersection(*map(set, [list(teacher_wave_dict[1].columns.values),
                                                                    list(teacher_wave_dict[2].columns.values),
                                                                    list(teacher_wave_dict[3].columns.values),
                                                                    list(teacher_wave_dict[4].columns.values),
                                                                    list(teacher_wave_dict[5].columns.values),
                                                                    list(teacher_wave_dict[6].columns.values)
                                                                    ])))

        new_df = pd.DataFrame(columns=teacher_constructs_in_all_6_list)
        unique_cases = set()
        # get a complete teacher dataframe for descriptives
        for k, df in teacher_wave_dict.items():
            df = df[teacher_constructs_in_all_6_list]
            id_col = df["clnr"]
            # add the column values to the set of unique cases
            unique_cases.update(id_col.tolist())
            is_unique = id_col.isin(unique_cases)
            # filter the data frame and append rows with unique values to the new data frame
            unique_rows = df[is_unique]
            new_df = new_df.append(unique_rows)

        # reset the index of the new data frame
        complete_teacher_df = new_df.reset_index(drop=True)
        sext = complete_teacher_df['sext'].value_counts()
        print('teacher sex descriptives: ' + str(sext))

        pd.DataFrame(teacher_constructs_in_all_6_list).to_csv(teacher_save_path + "Teacher_cols_all_6.csv")
        print('additional teacher variables in all 6 waves, N = {}'.format(len(teacher_constructs_in_all_6_list)))

    # save all col list
    all_cols = pd.DataFrame(df_all.columns.values)
    all_cols.to_csv("Data/MetaData/all_columns.csv")

    # before impute, identify school track change:
    if school_track_change_var == True:
        switch_track_1 = []
        switch_track_2 = []
        switch_track_3 = []
        switch_track_4 = []
        switch_track_5 = []
        switch_track_6 = []
        for index, row in df_all.iterrows():
            if (row[['sctyp_1', 'sctyp_2']].isnull().any() == False) and (row['sctyp_1'] != row['sctyp_2']) or \
                (row[['sctyp_2', 'sctyp_3']].isnull().any() == False) and (row['sctyp_2'] != row['sctyp_3'])or \
                (row[['sctyp_3', 'sctyp_4']].isnull().any() == False) and (row['sctyp_3'] != row['sctyp_4'])or \
                (row[['sctyp_4', 'sctyp_5']].isnull().any() == False) and (row['sctyp_4'] != row['sctyp_5'])or \
                (row[['sctyp_5', 'sctyp_6']].isnull().any() == False) and (row['sctyp_5'] != row['sctyp_6']):
                # print('switch track!')
                # print(row[['sctyp_2', 'sctyp_3', 'sctyp_4', 'sctyp_5', 'sctyp_6']])
                switch_track_1.append(1)
                switch_track_2.append(1)
                switch_track_3.append(1)
                switch_track_4.append(1)
                switch_track_5.append(1)
                switch_track_6.append(1)
            else:
                # print("doesn't switch track!")
                # print(row[['sctyp_2', 'sctyp_3', 'sctyp_4', 'sctyp_5', 'sctyp_6']])
                switch_track_1.append(0)
                switch_track_2.append(0)
                switch_track_3.append(0)
                switch_track_4.append(0)
                switch_track_5.append(0)
                switch_track_6.append(0)
        df_all['switch_track_1'] = switch_track_1
        df_all['switch_track_2'] = switch_track_2
        df_all['switch_track_3'] = switch_track_3
        df_all['switch_track_4'] = switch_track_4
        df_all['switch_track_5'] = switch_track_5
        df_all['switch_track_6'] = switch_track_6

    # Fill sctyp nas (allowing for change in school track to occur)
    df_all[['sctyp_2', 'sctyp_3', 'sctyp_4', 'sctyp_5', 'sctyp_6']].bfill(axis=1).ffill(axis=1)

    # Fill aspds_1 with aspds_2:
    df_all['aspds_1'] = df_all["aspds_2"].copy()

    # Sort epg (SES vars): egp6_1 for w1 and w2, egp6_3 for w3 and w4, and egp6_5 for w5 and w6
    df_all[['egp_1', 'egp_3', 'egp_5']] = df_all[['egp6_1', 'egp6_3', 'egp6_5']].ffill(axis=1).bfill(axis=1)
    df_all['egp_2'] = df_all['egp_1'].copy()
    df_all['egp_4'] = df_all['egp_3'].copy()
    df_all['egp_6'] = df_all['egp_5'].copy()
    df_all.drop(['egp6_1', 'egp6_3', 'egp6_5'], axis=1, inplace=True)

    # Fill in immigration proxies for time points not collected (most recent first, assumed constant)
    # in cases where there is inconsistency assume time 5 (most recent correct)
    if add_immigration == True:
        # where student born
        df_all[['borns_1', 'borns_3', 'borns_5']] = df_all[['borns_1', 'borns_3', 'borns_5']].bfill(axis=1).ffill(axis=1)
        df_all['borns_2'] = df_all['borns_5']
        df_all['borns_4'] = df_all['borns_5']
        df_all['borns_6'] = df_all['borns_5']
        # incase 1 or 3 not consistent with 5, take 5 to ensure constant:
        df_all['borns_1'] = df_all['borns_5']
        df_all['borns_3'] = df_all['borns_5']

        # where mother born
        df_all[['bornm_1', 'bornm_3', 'bornm_5']] = df_all[['bornm_1', 'bornm_3', 'bornm_5']].bfill(axis=1).ffill(axis=1)
        df_all['bornm_2'] = df_all['bornm_5']
        df_all['bornm_4'] = df_all['bornm_5']
        df_all['bornm_6'] = df_all['bornm_5']
        # incase 1 or 3 not consistent with 5, take 5 to ensure constant:
        df_all['bornm_1'] = df_all['bornm_5']
        df_all['bornm_3'] = df_all['bornm_5']

        # where father born
        df_all[['bornf_1', 'bornf_3', 'bornf_5']] = df_all[['bornf_1', 'bornf_3', 'bornf_5']].bfill(axis=1).ffill(axis=1)
        df_all['bornf_2'] = df_all['bornf_5']
        df_all['bornf_4'] = df_all['bornf_5']
        df_all['bornf_6'] = df_all['bornf_5']
        # incase 1 or 3 not consistent with 5, take 5 to ensure constant:
        df_all['bornf_1'] = df_all['bornf_5']
        df_all['bornf_3'] = df_all['bornf_5']

        # language spoken at home (can technically vary over time, allow this)
        df_all[['langu_1', 'langu_3', 'langu_5']] = df_all[['langu_1', 'langu_3', 'langu_5']].bfill(axis=1).ffill(axis=1)
        # impute with the answer from the year before:
        df_all['langu_2'] = df_all['langu_1']
        df_all['langu_4'] = df_all['langu_3']
        df_all['langu_6'] = df_all['langu_5']

        # recode to '1' (Germany) vs 'other':
        for var in ['borns', 'bornm', 'bornf', 'langu']:
            for num in ['1', '2', '3', '4', '5', '6']:
                col = var + "_{}".format(num)
                df_all[col][df_all[col] > 1] = 0

    # recalculate aus items:
    if recalc_aus == True:
        old_aus_vars = ["ausus_1", "ausus_2", "ausus_3", "ausus_4", "ausus_5", "ausus_6"]
        df_all.drop(old_aus_vars, axis=1, inplace=True)
        df_all["ausus_1"] = (df_all["ausus1_1"] + df_all["ausus2_1"]) / 2
        df_all["ausus_2"] = (df_all["ausus1_2"] + df_all["ausus2_2"]) / 2
        df_all["ausus_3"] = (df_all["ausus1_3"] + df_all["ausus2_3"]) / 2
        df_all["ausus_4"] = (df_all["ausus1_4"] + df_all["ausus2_4"]) / 2
        df_all["ausus_5"] = (df_all["ausus1_5"] + df_all["ausus2_5"]) / 2
        df_all["ausus_6"] = (df_all["ausus1_6"] + df_all["ausus2_6"]) / 2

    # Rename some incorrectly named features:
    if renaming == True:
        print("Renaming variables...")
        rename_dict = {}
        checks = pd.DataFrame(df_all.columns).sort_values(by=0)
        checks.to_csv("Data/MetaData/Columns_ordered.csv")
        for col in df_all.columns:
            if col == "ouex4r_2":
                rename_dict[col] = "outex4r_2"

            if col == "kft_iq_1":
                rename_dict[col] = "kftiq_1"

            if col == "wh10_6":
                rename_dict[col] = "wh_10_6"

            if col == "wh9_6":
                rename_dict[col] = "wh_9_6"
            #(assumed error in original data):
            if col == "gmazz_8":
                rename_dict[col] = "gma_zz8"

        df_all.rename(rename_dict, inplace=True, axis=1)
        # drop vars after manual inspection:
        drops_raw = ["kft_t_1", "gzw_sn_3", "sari_1", "sari_2", "gymzw1_4", "gymzw1_5", "gymzw1_6",
                     "gymzw2_4", "gymzw2_5", "gymzw2_6", "gymzw3_5", "gymzw3_6"]
        df_all.drop(drops_raw, inplace=True, axis=1)

    # decode participant id
    if decode_pp_id == True:
        print('Decoding participant IDs...')
        school_list = []
        pp_list = []
        class_list = []
        school_year_lst = []
        for row in df_all.iterrows():
            lst = list(str(int(row[1]['vpnr'])))
            # School number:
            if len(lst) == 8:
            # first two numbers are the school:
                school = lst[0:2]
                school = "".join(school)
                school = int(school)
            elif len(lst) == 7:
            # first number is the school:
                school = lst[0:1]
                school = int(school[0])
            else:
                print("vpnr is len {}".format(str(len(lst))))
                school = np.nan
            school_list.append(school)

            # Student/Participant number (last 2 digits):
            participant_number = lst[-2:]
            participant_number = "".join(participant_number)
            participant_number = int(participant_number)
            pp_list.append(participant_number)

            # Class number is next two digits from the back:
            class_number = lst[-4:-2:]
            class_number = "".join(class_number)
            class_number = int(class_number)
            class_list.append(class_number)

            # School year started is next two from the back (i.e. "05" = year 5 = first wave)
            school_year = lst[-6:-4:]
            school_year = "".join(school_year)
            school_year = int(school_year)
            school_year_lst.append(school_year)

        df_all["School_Code"] = school_list
        df_all["Participant_Num"] = pp_list
        df_all["Class_Code"] = class_list
        df_all["School_Year"] = school_year_lst
        new_vars_list = ["School_Code", "Participant_Num", "Class_Code", "School_Year"]


    # Keep constructs only:
    if do_extract_constructs == True:
        print("Extracting constructs...")
        df = extract_constructs_df(df_all)


    # Get list of general col names:
    if general_list == True:
        print("Creating list of general variables (without wave/item info)...")
        vars = []
        labs = []
        ex = []
        for col, lab in zip(column_labels.keys(), list(column_labels.items())):
            # remove what is after final _:
            var = col.rsplit('_', 1)[0]

            # remove number from end of string:
            var_s = ''.join([i for i in var if not i.isdigit()])

            # if final character is '_', then remove:
            if var_s[-1] == '_':
                var_s = var_s[:-1]
            vars.append(var_s)

            # remove anything after 'item' (upper or lower):
            if lab[1]:
                lab_s = re.split('item', lab[1], flags=re.IGNORECASE)[0]
                labs.append(lab_s)
            else:
                lab_s='No Description'
                labs.append(lab_s)

            # add raw example:
            ex.append(col)

        all_labs = pd.DataFrame(zip(vars, ex, labs))
        all_labs.columns = ['General_Variable_Code', 'Raw_Example', 'Variable_Label']
        unique_labs = all_labs.drop_duplicates(subset='General_Variable_Code')
        unique_labs.reset_index(inplace=True, drop=True)

        save_path = "Data/MetaData/Constructs/"
        unique_labs.to_csv(save_path+"General_Constructs.csv")

    # translate the constructs:

    if translate == True:
        from Functions.Translate import translator
        column_labs = list(meta.column_labels)
        column_names = list(meta.column_names)
        print("Translating column labels from German to English...")
        translated = translator(list=column_labs)
        trans_df = pd.DataFrame(zip(column_names, column_labs, translated))
        trans_df.columns = ['Name', 'Label', 'Trans']
        trans_df.to_csv("Data/MetaData/Trans.csv")

    # plot distributions of variables:

    if plot_original_IVs == True:
        print("Plotting IVs...")
        save_path = "Preprocess_Outputs/Initial_descriptives/IV_hists/"
        dtype_df = pd.read_csv("Data/MetaData/VariableView.csv")

        for col in df.columns:
            if col in column_labels.keys():
                label = column_labels[col]
            else:
                label = col
            if col in list(dtype_df["Name"]):
                if dtype_df["Type"][dtype_df["Name"] == col].values[0] == "Numeric":
                    plot_hist(save_name=col, x=df[col],
                              bins=10, save_path=save_path, title=label,
                              xlim=None, ylim=None, xlab=col.replace('_',' '), ylab="Frequency",
                              fig_size=(6,6),
                              save_name_exceptions=["vpnr", "repbs_1", "fambs_1", "fambp_1", "cont_3", "clnr",
                                                    "clnr_1", "clnr_2", "clnr_3", "clnr_4", "clnr_5", "clnr_6",
                                                    "sctyp_1", "sctyp_2", "sctyp_3", "sctyp_4", "sctyp_5", "sctyp_6"])
                else:
                    print("String")
            else:
                print("Data type not known, col not in variable view")

    # code all missing data the same:

    if missing == True:
        print("Re-coding missing data to be consistent...")

        missing_df = pd.read_csv("Data/MetaData/VariableView.csv")
        missing_dict = {}
        exceptions = ["kftiq_1", "aspnp_2", "aspdp_2", 'egp_1', 'egp_2', 'egp_3', 'egp_4', 'egp_5', 'egp_6', 'aspds_1',
                      "bornf_2", "bornf_4", "bornf_6", "borns_2", "borns_4", "borns_6", "bornm_2", "bornm_4", "bornm_6",
                      "langu_2", "langu_4", "langu_6", "switch_track_1", "switch_track_2", "switch_track_3",
                      "switch_track_4", "switch_track_5", "switch_track_6"]

        for construct in df[df.columns[~df.columns.isin(new_vars_list)]].columns:
            if construct in exceptions:
                # (don't have missing data dictionary for these additional features)
                missing_lab = [np.nan]
            else:
                print(construct)
                missing_lab = missing_df["Missing"][missing_df["Name"]==construct].values[0].split(",")

                # strip whitespace away:
                missing_lab = [i.strip() for i in missing_lab]

            # add to dictionary:
            missing_dict[construct] = missing_lab

        # change missing value (as it is in dict) to Na
        for col in df[df.columns[~df.columns.isin(new_vars_list)]].columns:
            replace_value_list = missing_dict[col]
            if len(replace_value_list) > 1:
                for replace_value in replace_value_list:
                    if '- HI' in replace_value:
                        replace_value_new = replace_value.split(' - HI')[0]

                        # if replace_value_new or higher, change:
                        df[col][df[col] > float(replace_value_new)] = np.nan
                        df[col][df[col] == replace_value_new] = np.nan
                        df[col][df[col] == float(replace_value_new)] = np.nan

                    else:
                        df[col][df[col] == replace_value[0]] = np.nan
                        df[col][df[col] == float(replace_value)] = np.nan
            else:
                replace_value = missing_dict[col][0]
                if replace_value != 'None':
                    df[col][df[col] == replace_value] = np.nan
                    df[col][df[col] == float(replace_value)] = np.nan

        # some rogue '99's still exist despite not being coded as missing, check these:
        colname='99s'
        df_check99 = check_val(df=df, value=99, colname= colname,
                                save_path="Data/Initial_Preprocess/Check_99s/",
                                save_name='first_check_99s')

        # unless starting with kft, replace 99s in these columns with na:
        replace_cols = df_check99[df_check99[colname]>0]
        for col in list(replace_cols.index):
            if col.startswith('kft'):
                continue
            else:
                df[col] = df[col].replace(to_replace=99, value=np.nan)
                # additional recoding of missing data when 0 in these columns (should only be 1 or 2):
                if (col.startswith('vorp')) or (col.startswith('rueckp')):
                    df[col][df[col] == 0] = None

        df_check99 = check_val(df=df, value=99, colname= colname,
                               save_path="Data/Initial_Preprocess/Check_99s/",
                               save_name='second_check_99s')

        # Produce descriptives on missingness (counts and percentages) for each var
        missing_sum = pd.DataFrame(df.isna().sum())
        missing_perc = round((missing_sum / len(df)) * 100,2)
        missing_summary = pd.concat([missing_sum, missing_perc], axis=1)
        missing_summary.reset_index(inplace=True)
        missing_summary.columns = ["Variable", "Missing_Sum", "Missing_Perc"]

        # Add in english trans and export as csv
        eng_trans = pd.read_csv("Data/MetaData/Constructs/General_Constructs_EngTrans.csv")
        # join missing_summary to all_labs on all_labs["RawExample'] == missing_summary["Variable"]

        # Order by missingness
        save_path= "Preprocess_Outputs/Initial_descriptives/Missing_data_plots/"
        missing_summary.sort_values(by='Missing_Perc', ascending=False)

        # Plot missing perc as a hist
        plot_hist(save_path=save_path,
                  save_name="Missing_perc",
                  fig_size=(5,5),
                  bins=50,
                  fontsize=12,
                  x=missing_summary["Missing_Perc"],
                  title="Missing Data Percentages for Construct Variables",
                  xlab="Missing Data Percentage (%)",
                  ylab="Frequency")

        # save unique constructs df for analysis:
        data_save_path = "Data/"
        df.to_csv(data_save_path + "All_Constructs_Wide.csv")


    if plot_missing_data_by_wave == True:
        print("Extracting wave/grade info...")
        # First extract wave, grade, and var information:
        wave_dict = {}
        grade_dict = {}
        var_dict = {}
        drop_grade_list = []
        for col in df.columns:
            # extract what is after final _
            var = col.split("_")[-1]
            # if a 'jz' extract as grade:
            grade_string = re.findall(r'z\d+', var)
            if len(grade_string) == 1:
                grade = re.findall(r'\d+', grade_string[0])
            elif len(grade_string)>1:
                print("error: 2 grades!")
                grade = [np.nan]
            else:
                grade = [np.nan]
            grade_dict[col] = grade[0]

            # drop 'zz' grades due to repetition:
            drop_grade_string = re.findall(r'zz\d+', var)
            if len(drop_grade_string) == 1:
                drop_grade_list.append(col)

            # only if grade not detected, extract last digit as wave
            if (np.nan in grade) == True:
                # extract last number
                wave_number = re.findall(r'\d+', var)
                if len(wave_number) == 0:
                    wave_number = [0]
                if len(wave_number)>1:
                    print("error: 2 waves!")
                if wave_number[0]=="12345":
                    wave_number = [0]
                # save to dict:
                wave_dict[col]= wave_number[0]
            else:
                wave_dict[col] = 0

            # remove numbers after the final '_' in col:
            beg = col.rsplit('_', 1)[0]
            if len(col.rsplit('_', 1)) > 1:
                end = col.rsplit('_', 1)[1]
                end_letters = ''.join([i for i in end if not i.isdigit()])
                if len(end_letters) > 0:
                    var_nowave = beg + "_" + end_letters
                else:
                    var_nowave = beg
            else:
                var_nowave = beg
            var_dict[col] = var_nowave

        wave_df = pd.DataFrame.from_dict(wave_dict, orient="index").reset_index()
        wave_df.columns = ['Column', 'Wave']

        grade_df = pd.DataFrame.from_dict(grade_dict, orient="index").reset_index()
        grade_df.columns = ['Column2', 'Grade']

        var_df = pd.DataFrame.from_dict(var_dict, orient="index").reset_index()
        var_df.columns = ['Column3', 'Var']

        df_grade_wave = pd.concat([var_df, wave_df, grade_df], axis=1).drop(['Column2','Column3'], axis=1)
        df_grade_wave.to_csv("Data/MetaData/Waves_and_Grades.csv")

        # Long df:
        wave_df_dict = {}
        for wave_num in df_grade_wave['Wave'].unique():

            # drop variables from 'Var' column where in drop_grade_list (zz vars):
            df_grade_wave = df_grade_wave[~df_grade_wave['Var'].isin(drop_grade_list)]

            w_cols = df_grade_wave['Column'][df_grade_wave['Wave']==wave_num]
            if wave_num == 0:
                df_wave_num = df[w_cols]
                df_wave_num.set_index('vpnr', inplace=True, drop=True)
            if wave_num != 0:
                df_wave_num = df[['vpnr'] + list(w_cols)]
                df_wave_num.rename(columns=var_dict, inplace=True)
            wave_df_dict[wave_num] = df_wave_num
            # at this point DV is in save wave/year as IVs

        # merge teacher variables:
        if add_teacher_vars == True:
            for wave in range(1, 7):
                wave_df_dict[str(wave)] = wave_df_dict[str(wave)].merge(teacher_wave_dict[wave],
                                                                        on=["clnr"], how='left')
            print("teacher data merged")

        # impute teacher data across waves if same teacher id (so all class info is consistent)
        for df1, df2 in zip([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]):
            if (any(e in list(teacher_wave_dict[df2]['clnr']) for e in list(teacher_wave_dict[df1]['clnr'])) == True) or \
                    (any(e in list(teacher_wave_dict[df1]['clnr']) for e in list(teacher_wave_dict[df2]['clnr'])) == True):
                print('Crossover between df{} and df{}'.format(df1, df2))

    # Which features consistent over all a) 5 waves, b) 6 waves?
    if find_consistent_constructs == True:
        waves = [4, 5, 6]
        print("Finding constructs that are consistent over 5/6 waves...")
        save_path= "Data/MetaData/Constructs/"
        for wave in waves:
            if wave == 4:
                constructs_in_all_4_list = list(set.intersection(*map(set,[list(wave_df_dict['1'].columns.values),
                                                                    list(wave_df_dict['2'].columns.values),
                                                                    list(wave_df_dict['3'].columns.values),
                                                                    list(wave_df_dict['4'].columns.values),
                                                                    list(wave_df_dict['5'].columns.values)
                                                                     ])))
                constructs_in_all_4 = pd.DataFrame(constructs_in_all_4_list)
                # join labels and Eng translations:
                labels = pd.read_csv("Data/MetaData/Constructs/General_Constructs_EngTrans.csv".format(wave))
                constructs_in_all_4=pd.merge(left=constructs_in_all_4, right=labels, left_on=constructs_in_all_4[0],
                                             right_on="General_Variable_Code", how="left").iloc[:, 2:]
                constructs_in_all_4.to_csv(save_path+"Constructs_in_{}_waves.csv".format(wave))

            if wave == 5:
                constructs_in_all_5_list = list(set.intersection(*map(set,[list(wave_df_dict['1'].columns.values),
                                                                    list(wave_df_dict['2'].columns.values),
                                                                    list(wave_df_dict['3'].columns.values),
                                                                    list(wave_df_dict['4'].columns.values),
                                                                    list(wave_df_dict['5'].columns.values)
                                                                     ])))
                constructs_in_all_5 = pd.DataFrame(constructs_in_all_5_list)
                # join labels and Eng translations:
                labels = pd.read_csv("Data/MetaData/Constructs/General_Constructs_EngTrans.csv".format(wave))
                constructs_in_all_5=pd.merge(left=constructs_in_all_5, right=labels, left_on=constructs_in_all_5[0],
                                             right_on="General_Variable_Code", how="left").iloc[:, 2:]
                constructs_in_all_5.to_csv(save_path+"Constructs_in_{}_waves.csv".format(wave))

            if wave == 6:
                labels = pd.read_csv("Data/MetaData/Constructs/General_Constructs_EngTrans.csv")
                constructs_in_all_6_list = list(set.intersection(*map(set,[list(wave_df_dict['1'].columns.values),
                                                                    list(wave_df_dict['2'].columns.values),
                                                                    list(wave_df_dict['3'].columns.values),
                                                                    list(wave_df_dict['4'].columns.values),
                                                                    list(wave_df_dict['5'].columns.values),
                                                                    list(wave_df_dict['6'].columns.values)
                                                                     ])))
                constructs_in_all_6 = pd.DataFrame(constructs_in_all_6_list)
                constructs_in_all_6 = pd.merge(left=constructs_in_all_6, right=labels, left_on=constructs_in_all_6[0],
                                             right_on="General_Variable_Code", how="left").iloc[:, 2:]
                constructs_in_all_6.to_csv(save_path + "Constructs_in_{}_waves.csv".format(wave))

        additional_vars_in_5waves = [item for item in constructs_in_all_6_list if item not in constructs_in_all_5_list]
        if len(additional_vars_in_5waves) > 0:
            print("additional vars in 5 waves:")
            print(additional_vars_in_5waves)

        additional_vars_in_4waves = [item for item in constructs_in_all_5_list if item not in constructs_in_all_4_list]
        if len(additional_vars_in_4waves) > 0:
            print("additional vars in 4 waves:")
            print(additional_vars_in_4waves)

        # to drop after manual inspection (mostly subscales of overall constructs, gradex dropped as too similar to DV):
        drops_from_notes = ["clnr",
                            # ^meta, recalculated
                            "gradex", "fges",
                            # ^too similar DV
                            "comps", "decla", "ts_mzp", "mopa",
                            # ^see Notes,
                            "kftviq", "kftniq",
                            # ^taking the mean of verbal and non-verbal IQ (kftiq)
                            # rest are subscales:
                            "inters", "interf",
                            "shc", "shl", "sht",
                            "kftnra", "kftnt", "kftvra", "kftvt",
                            "bol", "boc",
                            "axt", "axl", "axc",
                            "joc", "jol", "jot",
                            "prt", "prl", "prc",
                            "agt", "agl", "agc",
                            "gma_zz5", "gde_zz5", "gla_zz5", "gsp_zz5",
                            # ^same as jz vars
                            'gphy_jz8', 'gphy_jz9']
                            # ^these features only exist grade 8 and up

        for row in constructs_in_all_6.iterrows():
            if (row[1]['General_Variable_Code'] in drops_from_notes) == True:
                constructs_in_all_6.drop(row[0], axis=0, inplace=True)
        constructs_in_all_6.reset_index(inplace=True)
        constructs_in_all_6.to_csv(save_path+"Constructs_in_6_waves.csv")

        # ****************************************
        # update wave_df_dict with final vars:
        final_vars = list(set(constructs_in_all_6_list) - set(drops_from_notes))
        final_vars_save = "Data/Initial_Preprocess/"
        pd.DataFrame(final_vars).to_csv(final_vars_save + "final_variables.csv")

        # copy original wave dict, and update wave dict with final vars list
        original_wave_df_dict = dict(wave_df_dict)
        dv_dict = {}
        print("~~~~~~~ final wave dicts ~~~~~~~")
        for w_num, w_df in wave_df_dict.items():
            if (w_num != 0) == True:
                if check_DV_descriptives == True:
                    if w_num == '1':
                        for pred_years_ahead in [1, 2, 3, 4]:
                            # check sges correlations between waves:
                            y_num = int(w_num) + pred_years_ahead
                            x_grade = 5
                            y_grade = x_grade + pred_years_ahead
                            print("x= df{} (grade {}); y= df{} (grade {})".format(w_num, x_grade, y_num, y_grade))
                            plot_scatt(x=wave_df_dict[w_num][dependent_variable],
                                       y=wave_df_dict[str(y_num)][dependent_variable],
                                       save_path="Data/Initial_Preprocess/Check_cors/Predict_X_years_ahead/",
                                       save_name="Grade_{}_and_grade_{}".format(x_grade, y_grade),
                                       xlab= "Grade {}".format(x_grade), ylab="Grade {}".format(y_grade),
                                       fontsize=12)
                dv_dict[w_num] = pd.DataFrame(w_df[dependent_variable])
                w_df = w_df[final_vars]
                # remove individuals which have no dependent variable info
                w_df.dropna(axis=0, inplace=True, subset=[dependent_variable])
                # todo: could keep for now and then later only remove when no T2 DV (as doesn't matter if missing at T1)
                # ^ but still not sure I want to imput sges_T1, so fine for now to only use complete DV and DV_T1 indivs
                # update final wave dict
                wave_df_dict[w_num]=w_df
                print("Wave {}: students= {}, vars={}".format(w_num, w_df.shape[0], w_df.shape[1]))
                # save final wave dicts:
                save_wave_path = "Data/Waves/"
                w_df.to_csv(save_wave_path+"wave_{}.csv".format(w_num))

                # compute the inter-class correlation of achievement scores for residual scores after regressing out school track + classroom vars

                blocks = pd.read_csv("Data/MetaData/Variable_blocks/final_blocked_vars.csv")
                blocks = blocks.T
                blocks.columns = blocks.iloc[0]
                blocks = blocks.tail(-1)

                ivs = list(['sctyp']) + list(blocks['Student_reported_class_context']) + list(
                    blocks['Teacher_reported_classroom_context'])
                ivs = [element for element in ivs if str(element) != "nan"]

                # Create and fit the linear regression model --- 5->6 transformed
                X = w_df[['vpnr'] + ivs]
                y = w_df['sges']

                w_df_i = pd.concat([X, y], axis=1)
                w_df_i.dropna(how="any", inplace=True, axis=0)

                X = w_df_i.drop('sges', axis=1)
                y = w_df_i['sges']

                # get unique class id:
                class_ids = []
                for row in X.iterrows():
                    lst = list(str(int(row[1]['vpnr'])))
                    # Class number is next two digits from the back:
                    class_number = lst[-4:-2:]
                    class_number = "".join(class_number)
                    # School number:
                    if len(lst) == 8:
                        # first two numbers are the school:
                        school = lst[0:2]
                        school = "".join(school)
                    elif len(lst) == 7:
                        # first number is the school:
                        school = lst[0:1]
                        school = school[0]
                    else:
                        print("incorrect vpnr")
                        breakpoint()
                    # create School_class, a unique class identifier
                    school_class = school + "_" + class_number
                    class_ids.append(school_class)

                X['class_id'] = class_ids
                print("in wave {} there are {} unique classes".format(w_num, X.class_id.value_counts().shape[0]))

                X_m = X.copy()
                X_m.drop('vpnr', axis=1, inplace=True)
                X_m = X.loc[:, X.columns != 'class_id']

                model = LinearRegression()
                model.fit(X_m, y)
                # Predict the values
                y_pred = model.predict(X_m)
                # Calculate the residuals
                residuals = y - y_pred

                # Construct data
                groups = X.class_id
                data = pd.concat([residuals, groups, X['vpnr']], axis=1)
                data['vpnr'] = data['vpnr'].astype('object')

                # Calculate inter-class correlation using mixed lm (one student in only one class, doesn't matter if unequal number of students in each class)
                formula = 'sges ~ 1'

                # Fit the mixed effects model
                model = mixedlm(formula, data=data, groups='class_id', re_formula='1', missing='drop')
                result = model.fit()

                # Extract the ICC value
                icc = result.cov_re['class_id'] / (result.cov_re['class_id'] + result.scale)
                print('Wave {}, ICC: {}'.format(w_num, round(icc, 2)))

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if check_DV_descriptives == True:
            # check dv correlations:
            dv_compar_df = pd.concat(dv_dict.values(), ignore_index=True, axis=1)
            dv_compar_df.to_csv("Data/Initial_Preprocess/Check_cors/Predict_X_years_ahead/DVs_df/full.csv")
            # count Ns:
            print("DV Ns (non-NA): ")
            dv_ns = dv_compar_df.count()
            print(dv_ns)
            dv_ns.to_csv("Data/Initial_Preprocess/Check_cors/Predict_X_years_ahead/DVs_df/ns.csv")
            # cors:
            print("DV cors:")
            dv_cors = round(dv_compar_df.corr(method='pearson', min_periods=None), 2)
            print(dv_cors)
            dv_cors.to_csv("Data/Initial_Preprocess/Check_cors/Predict_X_years_ahead/DVs_df/corrs.csv")
        # ****************************************

        # Compose a list of constructs which are not consistent across all waves (for checking purposes):
        not_cons = unique_labs['General_Variable_Code'][
            ~unique_labs['General_Variable_Code'].isin(list(constructs_in_all_6['General_Variable_Code'])+['vpnr'])]
        not_cons = not_cons.reset_index()
        not_cons.rename(columns={'General_Variable_Code':'Constructs_not_in_all_waves'}, inplace=True)
        not_cons.to_csv("Data/MetaData/Constructs/Constructs_not_in_all_waves.csv")

        # Take out time consistent (invariant) variables from list
        wave_0_general = []
        for i in list(wave_df_dict[0].columns):
            # remove after the final '-':
            general = i.rsplit('_', 1)[0]
            wave_0_general.append(general)

        not_cons2 = not_cons['Constructs_not_in_all_waves'][
            ~not_cons['Constructs_not_in_all_waves'].isin(wave_0_general + ['vpnr'])]
        not_cons2 = not_cons2.reset_index()
        # drop after manual inspection
        drops_from_non_matched_list = ["varsr", "scnr", "fvalsr", "fvalpr", "gconr", "modepr",
                                       "fvapr", "aususa", "aususk", "ausupa"]
        for row in not_cons2.iterrows():
            if (row[1]['Constructs_not_in_all_waves'] in drops_from_non_matched_list) == True:
                not_cons2.drop(row[0], axis=0, inplace=True)
        not_cons2.reset_index(inplace=True)
        # save list
        not_cons2.rename(columns={'Constructs_not_in_all_waves': 'Constructs_not_in_all_waves_TimeInconsistentRemoved'}, inplace=True)
        not_cons2.to_csv("Data/MetaData/Constructs/Constructs_not_in_all_waves_TimeInconsistentRemoved.csv")

    if plot_missing_data_by_wave == True:
        print("Plotting wave missing data summaries...")
        wave_list = []
        N_col_list = []
        nonna_df_list = []
        dv_count_list = []

        for wave_number, data_frame in wave_df_dict.items():
            data_frame.reset_index()
            n_cols=len(data_frame.columns)
            count_nonna = data_frame.count(axis=0)
            perc_nonna = round((data_frame.count(axis=0)/len(data_frame))*100,2)
            nona_df = pd.DataFrame(pd.concat([count_nonna, perc_nonna], axis=1)).reset_index()
            nona_df.columns = ["Variable", "Number_Students", "Percentage_Students"]
            # Add labels / eng trans in:
            labels = pd.DataFrame.from_dict(column_labels, orient='index').reset_index()
            nona_df = pd.merge(left=nona_df, right=labels, left_on=nona_df['Variable'],
                     right_on="index", how="left").drop(labels='index', axis=1)
            nona_df.to_csv(save_path+"Wave_{}_Missing.csv".format(str(wave_number)))
            # Save list of Time constant variables:
            if wave_number == 0:
                construct_save_path = "Data/MetaData/Constructs/"
                time_constant_missing = nona_df.copy()
                time_constant_missing.to_csv(construct_save_path+"Time_Constant_Constructs.csv")

            if by_wave_plotting == True:
                # DV:
                if wave_number != 0:
                    count_dv = data_frame[dependent_variable].count()

                    # plot a hist":
                    save_path = "Analysis_1_with_sges_T1/Outputs/Descriptives/Missing_Data/"
                    plot_hist(save_path=save_path,
                              save_name="Wave_{}_Maths_Mark_{}".format(str(wave_number), dependent_variable),
                              fig_size=(5, 5),
                              bins=50,
                              fontsize=12,
                              x=data_frame[dependent_variable],
                              title="Maths Mark ({}) Distribution for Wave {}".format(dependent_variable, str(wave_number)),
                              xlab="Maths Mark ({})".format(dependent_variable),
                              ylab="Frequency")
                else:
                    count_dv = np.nan

                # plot IVs N per wave:
                save_path = "Preprocess_Outputs/Initial_descriptives/Missing_data_plots/"
                print('Plotting N students')
                plot_bar_h_df(y="Number_Students", x="Variable", df=nona_df,
                         xlab="Number of students with data (the rest is missing)", ylab="Variable", save_path=save_path,
                         title="Wave {} Variables".format(str(wave_number)),
                         save_name="Wave_{}_N_Students".format(str(wave_number)))

                print('Plotting perc students')
                plot_bar_h_df(y="Percentage_Students", x="Variable", df=nona_df,
                         xlab="Percentage of students with data", ylab="Variable", save_path=save_path,
                         title="Wave {} Variables".format(str(wave_number)),
                         save_name="Wave_{}_Perc_Students".format(str(wave_number)))

                wave_list.append(wave_number)
                N_col_list.append(n_cols)
                nonna_df_list.append(nona_df)
                dv_count_list.append(count_dv)

        if by_wave_plotting == True:
            plot_bar_h(x=np.array(wave_list), y=np.array(dv_count_list),
                       ylab="Wave", xlab="Number of Students", save_path=save_path,
                       title="Maths Mark ({}) N Students".format(dependent_variable),
                       save_name="DV_Maths_Mark_({})_N_Students".format(dependent_variable))


    # create 5 different long dataframes for analysis (wave 1+2, 2+3 etc.)
    if create_anal_df == True:
        print('Creating analysis dataframes...')
        load_path = "Data/Initial_Preprocess/"
        overlap_cols = pd.read_csv(load_path+"final_variables.csv")
        overlap_cols = list(overlap_cols['0'])

        # create separate df for time constant vars:
        data_path = "Data/"
        construct_save_path = "Data/MetaData/Constructs/"
        time_constant_missing = pd.read_csv(construct_save_path + "Time_Constant_Constructs.csv")
        time_constant_vars = list(time_constant_missing["Variable"])

        # drop zz grades:
        wave_0 = wave_df_dict[0]
        time_constant_df = wave_0[time_constant_vars]
        time_constant_df.drop(drop_grade_list + ['gphy_jz8', 'gphy_jz9'], inplace=True, axis=1)
        pd.DataFrame(drop_grade_list).to_csv("Data/MetaData/drop_grade_list.csv")

        # save time constant vars
        time_constant_df.to_csv(data_path+"Time_Constant_All.csv")

        # add wave col to each df
        for wave in [1, 2, 3, 4, 5, 6]:
            df = wave_df_dict[str(wave)]
            df['Wave'] = str(wave)
            wave_df_dict[str(wave)] = df

        # add any other cols we want to list here:
        cols = ['Wave'] + overlap_cols
        df_1 = pd.concat([wave_df_dict['1'][cols], wave_df_dict["2"][cols]], axis=0)
        df_1.sort_values(by=['vpnr','Wave'], inplace=True)

        df_2 = pd.concat([wave_df_dict['2'][cols], wave_df_dict["3"][cols]], axis=0)
        df_2.sort_values(by=['vpnr','Wave'], inplace=True)

        df_3 = pd.concat([wave_df_dict['3'][cols], wave_df_dict["4"][cols]], axis=0)
        df_3.sort_values(by=['vpnr','Wave'], inplace=True)

        df_4 = pd.concat([wave_df_dict['4'][cols], wave_df_dict["5"][cols]], axis=0)
        df_4.sort_values(by=['vpnr','Wave'], inplace=True)

        df_5 = pd.concat([wave_df_dict['5'][cols], wave_df_dict["6"][cols]], axis=0)
        df_5.sort_values(by=['vpnr','Wave'], inplace=True)

        # only keep indivdiuals where in both waves...
        data_path = "Data/"
        dfs = [df_1, df_2, df_3, df_4, df_5]
        waves = [1, 2, 3, 4, 5]
        print('Ns in dataframes', file=open("Preprocess_Outputs/Modelling_dfs/df_Ns/dfNs.txt", "w"))

        for wdf, wave in zip(dfs, waves):
            wdf['col_count'] = wdf.apply(lambda x: x.count(), axis=1)
            wdf = wdf[wdf['col_count'] > 5]
            n = len(wdf)
            wa_n = len(wdf[wdf['Wave'] == str(wave)])
            wb_n = len(wdf[wdf['Wave'] == str(wave + 1)])

            # see how many unique individuals in both waves with complete info:
            wdf["Wave"] = wdf["Wave"].astype(int)
            # count where count==2 (have data from both wave 1 and 2)
            counts = wdf.loc[:, ['Wave', 'vpnr']].groupby("vpnr").count()
            counts = counts[counts["Wave"] == 2]
            list_comp_ids = list(counts.index)

            # only keep rows where ids in both waves:
            w_dframe_red = wdf[wdf["vpnr"].isin(list_comp_ids)]
            w_dframe_red.drop("col_count", axis=1, inplace=True)
            w_dframe_red.to_csv(data_path + 'Modelling/df{}_complete.csv'.format(wave))
            n_comp = len(w_dframe_red["vpnr"].unique())
            print('Dataframe {} (wave: {} -> {}),'
                  'N= {}; wave {} = {}, wave {} = {}. '
                  'Unique IDs in both waves: {}'.format(wave, wave, wave+1, n, wave, wa_n, wave + 1, wb_n, n_comp),
                  file=open("Preprocess_Outputs/Modelling_dfs/df_Ns/dfNs.txt", "a"))

    # Final steps to creating dataframes predicting one year ahead
    if final_steps == True:
        include_grade_4_vars = False
        data_path = 'Data/Modelling/'
        directory_bits = os.fsencode(data_path)
        missing_outputs_path = "Preprocess_Outputs/Modelling_dfs/Missingness/"
        save_path_df_info = "Preprocess_Outputs/Modelling_dfs/df_Ns/"

        # load time invariant data:
        df_time_constant = pd.read_csv("Data/Time_Constant_All.csv", index_col=False)
        # save aggregate level vars for later use:
        identifiers = ["vpnr", "School_Code", "Class_Code"]
        df_aggregate = df_time_constant[identifiers]
        df_aggregate.to_csv("Data/MetaData/Aggregate_Cols.csv")

        # # drop those not needed:
        # df_time_constant.drop(["School_Code", "Participant_Num", 'Class_Code', 'School_Year'],
        #                       axis=1, inplace=True)
        # todo: drop later

        # open doc to record droppped vars for each df:
        dropped_vars_missing_path = "Preprocess_Outputs/Modelling_dfs/Dropped_features_missing/"
        print("Dropped variables due to missing data... ",
              file=open(dropped_vars_missing_path + "Dropped_vars.txt", "w"))

        drop_var_list = []
        # create dataframes predicting 1 year ahead:
        for file in os.listdir(directory_bits):
            filename = os.fsdecode(file)
            if filename.endswith("_complete.csv"):
                print("--> Processing.... {}".format(filename))
                df_num = [s for s in re.findall(r'\d+', filename)][0]

                df = pd.read_csv(data_path+filename, index_col=[0])

                # define X and y:
                y = df["sges"][df["Wave"] == int(df_num)+1]
                if preprocess_drop_DV_T1 == True:
                    Xi = df.drop("sges", axis=1)[df["Wave"] == int(df_num)]
                if preprocess_drop_DV_T1 == False:
                    Xi = df[df["Wave"] == int(df_num)]
                    Xi.rename(columns={dependent_variable: dv_T1}, inplace=True)

                X = Xi.merge(df_time_constant, on="vpnr")

                # check if class_code can be re-identified by the teacher vars:
                block_dict = parse_data_to_dict(block_file)
                teacher_cols = block_dict['Teacher_reported_classroom_context']
                teacher_cols.remove('sext_2.0')
                teacher_cols.remove('aget')
                redo_class_id = X[identifiers + teacher_cols + ['sctyp']]
                redo_class_id['School_Class'] = redo_class_id['School_Code'].astype(str) + "_" + redo_class_id[
                    'Class_Code'].astype(str)
                count_class_ids = len(redo_class_id[['School_Class']].drop_duplicates())
                count_class_same_rows = len(redo_class_id[teacher_cols].drop_duplicates())
                print('num of class: {}; num of classes inferred by data values: {}'.format(count_class_ids, count_class_same_rows))
                if count_class_ids == count_class_same_rows:
                    print('class ids can be re-engineered from teacher vars :)')
                else:
                    print('class ids can NOT be re-engineered from teacher vars')

                # drop for now (Meta):
                drop_meta = ["Wave", "filter_$", "agemon", "ageyear", 'index'] +\
                            ["School_Code", "Participant_Num", 'Class_Code', 'School_Year']
                df = remove_cols(df, drop_meta)

                # only keep grade variables that allow forward time prediction:
                # go through time constant columns:
                drops_list = []
                keep_list = []
                for var in df_time_constant.columns:
                    if include_grade_4_vars == True:
                        # keep in (ignore) if a 4 (primary school grade) or ends with '12345':
                        if var.endswith('4') or var.endswith('12345'):
                            keep_list.append(var)
                        else:
                            continue
                    else:
                        # does end with digit?:
                        is_digit = var[-1].isdigit()
                        # if doesn't end with digit then continue
                        if is_digit == False:
                            continue
                        else:
                            grade_num = int(var[-1])
                            # only keep vars where final number is ==df_num+4 (grade IVs should be from)
                            if grade_num != int(df_num)+4:
                                drops_list.append(var)
                            else:
                                keep_list.append(var)
                # drop rest
                X.drop(drops_list, inplace=True, axis=1)

                # drop some IVs due to too much missingness:
                missing_sum = pd.DataFrame(X.isna().sum())
                missing_perc = round((missing_sum / len(X)) * 100, 2)
                missing_summary = pd.concat([missing_sum, missing_perc], axis=1)
                missing_summary.reset_index(inplace=True)
                missing_summary.columns = ["Variable", "Missing_Sum", "Missing_Perc"]
                missing_summary.sort_values(by='Missing_Perc', ascending=False, inplace=True)
                missing_summary.to_csv(missing_outputs_path + "Missing_data_summary_df{}.csv".format(df_num))

                # Plot missing perc as a hist
                plot_hist(save_path=missing_outputs_path,
                          save_name="Missing_perc_df{}".format(df_num),
                          fig_size=(5, 5),
                          bins=500,
                          fontsize=12,
                          x=missing_summary["Missing_Perc"],
                          title="Missing Data Percentages for Variables",
                          xlab="Missing Data Percentage (%)",
                          ylab="Frequency")

                # drop vars with missing data > cutoff (except for df5 when only use dv)
                if final_is_4_timepoints == True:
                    if (df_num == '4') or (df_num == '5'):
                        print('no vars dropped for missingness from df{} by design'.format(df_num))
                    else:
                        # only removes IV missingness for dfs 1-4
                        drop_vars_perc = missing_summary[["Variable", "Missing_Perc"]][
                            missing_summary["Missing_Perc"] >= missing_data_col_cutoff]
                        drop_vars = missing_summary["Variable"][
                            missing_summary["Missing_Perc"] >= missing_data_col_cutoff]
                        drop_var_list.append(list(drop_vars.values))
                        X.drop(drop_vars, axis=1, inplace=True)
                        print("DF{}: Dropped {} variables due to missing data. {} variables remaining\n"
                              "Dropped variables: {}".format(df_num, len(drop_vars), X.shape[1], drop_vars_perc),
                              file=open(dropped_vars_missing_path + "Dropped_vars.txt", "a"))

                if final_is_4_timepoints == False:
                # assume using all 5 time points (using df 6)
                    if df_num == '5':
                        print('no vars dropped for missingness from df{} by design'.format(df_num))
                    else:
                        # only removes IV missingness for dfs 1-4
                        drop_vars_perc = missing_summary[["Variable", "Missing_Perc"]][
                            missing_summary["Missing_Perc"] >= missing_data_col_cutoff]
                        drop_vars = missing_summary["Variable"][
                            missing_summary["Missing_Perc"] >= missing_data_col_cutoff]
                        drop_var_list.append(list(drop_vars.values))
                        X.drop(drop_vars, axis=1, inplace=True)
                        print("DF{}: Dropped {} variables due to missing data. {} variables remaining\n"
                              "Dropped variables: {}".format(df_num, len(drop_vars), X.shape[1], drop_vars_perc),
                              file=open(dropped_vars_missing_path + "Dropped_vars.txt", "a"))

                # merge and save X and y
                save_path = "Data/Initial_Preprocess/"
                y = pd.Series(y).reset_index()
                X_and_y = pd.concat([X, y], axis=1)

                # remove more vars
                drop_list = ["Unnamed: 0", "index", "laeng12345", "vpnrg_u", "agemon", "startc", "starty",
                             "laeng12345_C", "laeng12345_S", "filter_$", "ageyear", "Wave"]
                for col in X_and_y.columns:
                    if col in drop_list:
                        X_and_y.drop(col, axis=1, inplace=True)
                    # rename grade vars (remove number):
                    if any(i.isdigit() for i in col):
                        if col == "sges_T1":
                            continue
                        col_new = ''.join([i for i in col if not i.isdigit()])
                        X_and_y.rename({col:col_new}, inplace=True, axis=1)

                # Merge high cardinal feature categories
                X_and_y['houspan'][X_and_y["houspan"] > 9] = 999
                X_and_y['houschn'][X_and_y["houschn"] > 9] = 999

                # initial save
                X_and_y.to_csv(save_path + "df{}_preprocessed{}.csv".format(df_num, version))

        # if a var dropped for missingness on one df, drop on others
        drop_var_list_flat = list(itertools.chain(*drop_var_list))
        drop_var_list_flat = list(set(drop_var_list_flat))
        directory_bits = os.fsencode(save_path)
        for file in os.listdir(directory_bits):
            filename = os.fsdecode(file)
            if filename.endswith("_preprocessed{}.csv".format(version)):
                print("--> Processing... {}".format(filename))
                df_num = [s for s in re.findall(r'\d+', filename)][0]
                df = pd.read_csv(save_path+filename)
                for drop_col in drop_var_list_flat:
                    drop_col_start = drop_col.split('_')[0]
                    for col in df:
                        if str(col).startswith(drop_col_start):
                            df.drop(col, axis=1, inplace=True)
                        else:
                            continue

                print("df{} shape: {}".format(df_num, df.shape))
                df.to_csv(save_path + "df{}_preprocessed{}.csv".format(df_num, version))
                # save feature df:
                cols = pd.DataFrame(df.columns.values)
                col_save_path = "Data/Initial_Preprocess/Cols_lists/"
                cols.to_csv(col_save_path + "df{}_cols.csv".format(df_num))

                # print n and p for each df:
                save_file = "Descriptives_df{}".format(df_num)
                n = df.shape[0]
                p = int(df.shape[1]) - 1
                print("df {} descriptives :\nn = {}; p = {}".format(df_num, n, p),
                      file=open(save_path_df_info + save_file, "w"))

                # final check of IV cors
                if check_DV_descriptives == True:

                    # check sges correlations between DV_T1 and DV:
                    grade_IV_dict = {1:5, 2:6, 3:7, 4:8, 5:9}
                    x_grade = grade_IV_dict[int(df_num)]
                    y_grade = x_grade + 1
                    print("df{}: x= grade {}; y= grade {}".format(df_num, x_grade, y_grade))
                    plot_scatt(x=df[dv_T1],
                               y=df[dependent_variable],
                               save_path="Data/Initial_Preprocess/Check_cors/Main_anal/",
                               save_name="FINAL_Grade_{}_and_grade_{}".format(x_grade, y_grade),
                               xlab= "Grade {}".format(x_grade), ylab="Grade {}".format(y_grade),
                               fontsize=12)

                    if df_num == '1':
                        x_grade = 5
                        print("---------------- final checks ------------------")
                        for pred_years_ahead in [1, 2, 3, 4]:
                            # check sges correlations between waves:
                            df_x = df.copy()
                            if pred_years_ahead == 1:
                                y_num = df_num
                                y_grade = x_grade + pred_years_ahead
                                print("x= df{} (grade {}); y= df{} (grade {})".format(df_num, x_grade, y_num, y_grade))
                                print("N={}".format(df_x.shape[0]))
                                plot_scatt(x=df_x[dv_T1],
                                           y=df_x[dependent_variable],
                                           save_path="Data/Initial_Preprocess/Check_cors/Predict_X_years_ahead/",
                                           save_name="FINAL_Grade_{}_and_grade_{}".format(x_grade, y_grade),
                                           xlab="Grade {}".format(x_grade), ylab="Grade {}".format(y_grade),
                                           fontsize=12)
                            else:
                                y_num = pred_years_ahead
                                df_y = pd.read_csv(save_path + "df{}_preprocessed{}.csv".format(y_num, version))
                                y_grade = x_grade + pred_years_ahead
                                x = df_x[dv_T1]
                                y = df_y[dependent_variable]
                                print("x= df{} (grade {}); y= df{} (grade {})".format(df_num, x_grade, y_num, y_grade))
                                if x.shape[0] != y.shape[0]:
                                    join_df = df_x[['vpnr', dv_T1]].merge(df_y[['vpnr', dependent_variable]],
                                                                         how='inner', on="vpnr")
                                    x = join_df[dv_T1]
                                    y = join_df[dependent_variable]
                                    print("N={}".format(join_df.shape[0]))

                                # remove where both have NAs
                                plot_scatt(x=x,
                                       y=y,
                                       save_path="Data/Initial_Preprocess/Check_cors/Predict_X_years_ahead/",
                                       save_name="FINAL_Grade_{}_and_grade_{}".format(x_grade, y_grade),
                                       xlab="Grade {}".format(x_grade), ylab="Grade {}".format(y_grade),
                                       fontsize=12)
                            print("-----------")


    if plot_IVs_by_wave == True:
        data_path = "Data/Initial_Preprocess/"
        directory_bits = os.fsencode(data_path)
        save_path = "Preprocess_Outputs/Modelling_dfs/Plot_IVs/"
        for file in os.listdir(directory_bits):
            filename = os.fsdecode(file)
            if filename.endswith("_preprocessed{}.csv".format(version)):
                print("--> Plotting.... {}".format(filename))
                df_num = [s for s in re.findall(r'\d+', filename)][0]

                df = pd.read_csv(data_path+filename, index_col=[0])

                for col in df.columns:
                    plot_hist(save_name=col, x=df[col],
                              bins=10, save_path=save_path + "df{}/".format(df_num), title=col,
                              xlim=None, ylim=None, xlab=col.replace('_', ' '),
                              ylab="Frequency",
                              fig_size=(6, 6))

    # save end files:
    df_all.to_csv("Data/All.csv")
    df.to_csv("Data/df_constructs.csv")

    end_preprocessing = dt.datetime.now()
    preprocessing_time = end_preprocessing - start
    print("Preprocessing done. Time taken: {}".format(preprocessing_time))

