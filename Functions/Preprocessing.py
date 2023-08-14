import re
import pandas as pd

def check_val(df, value, colname, save_path, save_name):
    mask = df.isin([value])
    df_check = mask.sum().to_frame()
    df_check.columns = [colname]
    df_check.sort_values(ascending=False,
                         inplace=True,
                         axis=0,
                         by=[colname])
    df_check.to_csv(save_path + save_name + ".csv")
    return df_check


def multicollin_changes(df):

    df['pos_affect'] = (df['jo'] + df['pr']) / 2
    df['neg_affect'] = (df['ax'] + df['bo'] + df['ag'] + df['sh'] + df['hl']) / 5
    df['intrinsic_motiv'] = (df['mocom'] + df['moint']) / 2
    df['achieve_motiv'] = (df['mopav'] + df['mopap']) / 2
    df['self_efficacy'] = (df['acase'] + df['sefic'] + df['outex']) / 3
    # ^(see Marsh et al.,  murky distinction self-concept / self-efficacy, JEP)
    df['parental_expectation_degree'] = (df['aspns'] + df['aspds'] + df['aspnp'] + df['aspdp']) /4
    df['parental_expectation_math_grade'] = (df['expns'] + df['expds'] + df['expnp'] + df['expdp']) / 4
    df['parental_negative_aspiration'] = (df['exps'] + df['expp']) / 2
    df['model_behaviour'] = (df['modep'] + df['modes']) / 2
    df['instructional_behaviour'] = (df['instp'] + df['insts']) /2

    drops = ['ax', 'bo', 'ag', 'sh', 'hl', 'jo', 'pr', 'enval', 'acase', 'sefic', 'outex',
             'aspnp', 'aspdp', 'aspns', 'mocom', 'mopap', 'moint', 'mopav',
             'Unnamed: 0', 'Unnamed: 0.1', 'modep', 'modes', 'instp', 'insts',
             'expns', 'expds', 'expnp', 'expdp', 'exps', 'expp', 'aspds', 'aget'
    ]
    for col in df:
        if col in drops:
            df.drop(col, axis=1, inplace=True)

    return df

def extract_all_constructs(df):
    # filter out where {d}_
    regex = "^(.*?)\d"
    cols = list(df.columns)
    r = re.compile(regex)
    all_constructs = []
    for col in cols:
        if r.search(col):
            result = r.search(col)
            result = result.group(1)
            all_constructs.append(result)
    all_constructs = set(all_constructs)
    return all_constructs



def extract_constructs(df):
    # filter out where {d}_
    regex = "^((?![0-9]_).)*$"
    cols = list(df.columns)
    r = re.compile(regex)
    constructs = list(filter(r.match, cols))
    # also don't want if {d}r_
    regex = "^((?![0-9]r_).)*$"
    r = re.compile(regex)
    constructs = list(filter(r.match, constructs))
    # also don't want if {d}ra_
    regex = "^((?![0-9]ra_).)*$"
    r = re.compile(regex)
    constructs_list = list(filter(r.match, constructs))
    return constructs_list


def extract_constructs_df(df):
    constructs_list = extract_constructs(df)
    df_new = df[constructs_list]
    return df_new


def reaggregate_teacher_vars(df):
    constructs_list = extract_all_constructs(df)
    remove_list = ['id', 'page']
    constructs_list = [x for x in constructs_list if x not in remove_list]
    construct_dict = {}
    counter = 0
    for construct in constructs_list:
        construct_cols = []
        construct_name = construct
        counter = counter + 1
        for col in df:
            # find cols that start with construct:
            if col.startswith(construct):
                # if it contains a number...
                col_cleaned = col.split("_")[0]
                if any(map(str.isdigit, col_cleaned)):
                    # add to list
                    construct_cols.append(col)
                else:
                    # use as dict name
                    construct_name = col_cleaned

        if construct_name:
            construct_dict[construct_name] = construct_cols
        else:
            construct_name = "unknown_{}".format(counter)
            construct_dict[construct_name] = construct_cols

    reagg_df = get_mean_construct_from_dict(construct_dict, df)
    return reagg_df



def get_mean_construct_from_dict(dict, df):
    df_agg = pd.DataFrame()
    for key, item in dict.items():
        if len(item) > 1:
            df_agg[key] = df[item].mean(axis=1)
        else:
            non_agg_col = key + "_3"
            df_agg[key] = df[non_agg_col]
    id_var = pd.DataFrame(df['clnr'])
    df_agg = id_var.join(df_agg)
    return df_agg



def remove_reversed_teacher_vars(df):
    regex = "_r"
    cols = list(df.columns)
    r = re.compile(regex)
    removes = list(filter(r.search, cols))
    df.drop(removes, axis=1, inplace=True)
    return df



def rename_teacher_vars(df):
    vars = []
    for col in df.columns:
        # remove what is after final _:
        var = col.rsplit('_', 1)[0]
        vars.append(var)
    df.columns = vars
    return df


def remove_col(df, drop_col):
    for col in df:
        if col == drop_col:
            df.drop(col, inplace=True, axis=1)
    return df


def remove_cols(df, drop_cols):
    for col in df:
        if col in drop_cols:
            df.drop(col, inplace=True, axis=1)
    return df


def get_var_names_dict(df):
    dict = {}
    for index, row in df.iterrows():
        col_name = row["short_name"].split('(')[0]
        col_name = col_name.strip()
        long_name = row["long_name"]
        dict[col_name] = long_name
    return dict


def get_var_names_dict_agg(df, agg_level="class"):
    dict={}
    if agg_level == "class":
        ag = "_C"
    elif agg_level == "school":
        ag = "_S"
    else:
        ag = None
        print("enter either 'school' or 'class' as agg_level")
    for index, row in df.iterrows():
        col_name = row["short_name"].split('(')[0]
        col_name = col_name.strip()
        long_name = row["long_name"]
        dict[col_name] = long_name

        # agg level:
    for index, row in df.iterrows():
        col_name = row["short_name"].split('(')[0]
        col_name = col_name.strip()
        long_name = row["long_name"]
        col_name_agg = col_name + ag
        long_name_agg = long_name + " (C)"
        dict[col_name_agg] = long_name_agg
    return dict



def shap_dict_to_long_df(dict, cut_off, csv_save_path, csv_save_name):
    df_all = pd.concat(dict, axis=0)
    df_all["Abs_SHAP"] = abs(df_all["value"])
    # aggregate over people
    group = df_all[["variable", "Time", "Abs_SHAP"]]\
        .groupby(["variable", "Time"]).sum().reset_index()
    # aggregate over the time points
    var_agg = group[["variable", "Abs_SHAP"]].groupby(["variable"])\
        .sum().sort_values(by="Abs_SHAP", ascending=False)
    # save aggregate
    var_agg.to_csv(csv_save_path + csv_save_name)
    # only keep vars over an aggregate threshold
    var_agg = var_agg[var_agg["Abs_SHAP"] > cut_off]
    var_list = var_agg.index.to_list()
    subgroup = group[group["variable"].isin(var_list)]\
        .sort_values(by="Abs_SHAP", ascending=False)
    return subgroup



def find_cat_vars(meta):
    var_to_label_dict = meta.variable_to_label
    labels_dict = meta.variable_value_labels
    construct_labels_dict = {}
    for var, labels in labels_dict.items():
        # remove what is after final _:
        var = var.rsplit('_', 1)[0]

        # remove number from end of string:
        var_s = ''.join([i for i in var if not i.isdigit()])

        # if final character is '_', then remove:
        if var_s[-1] == '_':
            var_s = var_s[:-1]

        construct_labels_dict[var_s] = labels

    # remove reverse scored constructs
    del_key_list = []
    for construct, labels in construct_labels_dict.items():
        if str(construct).endswith('r'):
            # only remove if non-reversed version exists
            if str(construct)[:-1] in construct_labels_dict:
                del_key_list.append(construct)

    for key in del_key_list:
        del construct_labels_dict[key]

    return construct_labels_dict

