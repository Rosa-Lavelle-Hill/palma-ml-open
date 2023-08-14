import pandas as pd
from Fixed_params import cat_vars
from Functions.Block_Permutation_Importance import parse_data_to_dict
from Functions.Plotting import plot_cat_scatt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, threshold):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    au_corr = au_corr[au_corr>=threshold]
    return round(au_corr, 2)


def get_top_DV_abs_correlations(df, threshold, DV="sges"):
    au_corr = df.corr().abs()[DV].sort_values(ascending=False)
    au_corr = au_corr[au_corr>=threshold]
    return round(au_corr, 2)


def check_imputed_cors_across_blocks(X, analysis_path, block_file,
                                     corr_print_threshold,
                                     df, df_num, pipe,
                                     model, preprocessor, imp_method):

    numeric_features = X.drop(cat_vars, inplace=False, axis=1).columns
    dum_names = list(pipe.named_steps['preprocessor'].transformers_[1][1].named_steps['oh_encoder']
                     .get_feature_names(cat_vars))
    num_names = list(df[numeric_features].columns.values)
    names = num_names + dum_names

    # get imputed data:
    preprocessor.fit(X)
    X = preprocessor.transform(X)

    top_corrs = get_top_abs_correlations(pd.DataFrame(X, columns=names),
                                         threshold=corr_print_threshold).reset_index()
    # todo: only do for numerics features?^

    block_dict = parse_data_to_dict(block_file)
    check_blocks_dict = {}
    cor_save_path = "{}Outputs/Descriptives/Correlations/".format(analysis_path)
    for block_name, block_vars in block_dict.items():
        if len(block_vars) > 1:
            for index, row in top_corrs.iterrows():
                if ((row['level_0'] in block_vars) and (row['level_1'] not in block_vars)) \
                        or ((row['level_0'] not in block_vars) and (row['level_1'] in block_vars)):
                    cor = row[0]
                    offending_vars = [row['level_0'], row['level_1']]
                    offending_vars_string = str(offending_vars[0]) + ', ' + str(offending_vars[1])
                    check_blocks_dict[offending_vars_string] = cor
    #                 todo: prevent saving/remove duplicate other way around
    if check_blocks_dict != {}:
        block_cors_df = pd.DataFrame.from_dict(check_blocks_dict, orient="index")
        block_cors_df.reset_index(inplace=True)
        block_cors_df.columns = ["vars", "abs_cor"]
        block_cors_df.sort_values(by="abs_cor", ascending=False, inplace=True)

        # todo: add translation col names
        block_cors_df.to_csv(cor_save_path + "IMPUTED_{}Data_{}ImpModel_between_block_cors_above_{}_df{}.csv"
                             .format(model, imp_method, corr_print_threshold, df_num))
        print("-------------- {}, correlation above {} :---------------".format(model, cor))
        print(block_cors_df[0:21])



def eval_imputed_track_cors(X_train, analysis_path, df_num, cont_vars, var_names, X_train_orig,
                            xticks):

    log_reg = LogisticRegression(random_state=93, penalty='none', class_weight="balanced", max_iter=500)
    multi_log_reg = LogisticRegression(random_state=93, penalty='none', class_weight="balanced", max_iter=500,
                                       multi_class='multinomial')

    for var, var_string, xtick in zip(cont_vars, var_names, xticks):
        print(" __________________________ df: " + df_num)
        for num, name in zip(["3.0", "2.0", "1.0"],
                                         ["Hauptschule", "Realshule", "Gymnasium"]):
            school_type_num = "sctyp_" + num
            y_t = pd.DataFrame(X_train[school_type_num])
            print("predicting sctyp {}".format(school_type_num))
            print(y_t.value_counts())
            d = pd.concat([pd.DataFrame(X_train[var]),
                           y_t], axis=1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  binary:

            # point corr.
            d_int = d.copy()
            d_int[school_type_num].astype(int)
            cor = round(d_int.corr().iloc[0, 1], 2)

            log_reg.fit(pd.DataFrame(X_train[var]), y_t)
            y_pred = log_reg.predict(y_t)

            f1 = round(metrics.f1_score(y_t, y_pred, average="weighted"), 2)
            auc = round(metrics.roc_auc_score(y_t, y_pred, average="weighted"), 2)
            bal_acc = round(metrics.accuracy_score(y_t, y_pred), 2)
            print(metrics.classification_report(y_t, y_pred))
            title = "Df{}: Corr= {}; Log.Reg. F1= {}, AUC= {}, Acc= {}".format(df_num, cor, f1, auc, bal_acc)

            print(title)
            plot_cat_scatt(x=var, y=school_type_num, data=d, title=title, name=name,
                           ylab="School type: {}".format(name), xlab=var_string, xticks=xtick,
                           save_path=analysis_path + "Outputs/Descriptives/Correlations/Sctyp_Checks/"
                           , save_name="{}_cor_check_{}_df{}".format(var, school_type_num, df_num))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass:
        d = pd.concat([X_train_orig['sctyp'].reset_index(drop=True), X_train[var].reset_index(drop=True)], axis=1, ignore_index=True)
        d.columns=['sctyp', var]
        y_t = pd.DataFrame(d['sctyp'])

        # ANOVA:
        from scipy import stats
        F, p = stats.f_oneway(d[d['sctyp'] == 1.0][var],
                              d[d['sctyp'] == 2.0][var],
                              d[d['sctyp'] == 3.0][var])
        F = round(F, 0)
        p = round(p, 5)
        multi_log_reg.fit(pd.DataFrame(X_train[var]), y_t)
        y_pred = multi_log_reg.predict(y_t)
        f1 = round(metrics.f1_score(y_t, y_pred, average="weighted"), 2)
        # auc = round(metrics.roc_auc_score(sctyp_dum, y_pred, average="weighted", multi_class="ovo"), 2)
        bal_acc = round(metrics.accuracy_score(y_t, y_pred), 2)
        print(metrics.classification_report(y_t, y_pred))
        title = "Df{}: ANOVA F={}, p={}; Log.Reg. F1= {}, Acc= {}".format(df_num, F, p, f1, bal_acc)
        print(title)
        plot_cat_scatt(x=var, y='sctyp', data=d, title=title, xticks=xtick,
                       ylab="School Track", xlab=var_string, binary=False,
                       save_path=analysis_path + "Outputs/Descriptives/Correlations/Sctyp_Checks/"
                       , save_name="{}_cor_check_multiclass_df{}".format(var, df_num))