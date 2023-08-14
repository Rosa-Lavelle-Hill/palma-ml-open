import pandas as pd
import numpy as np
from Functions.Plotting import plot_results
from RUN_Predict_X_years_ahead_New import start_string, global_seed, analysis_path,\
    fixed_dummy_r2, Additional_pred_checks
from Fixed_params import Compare_DV_T1, mean_baseline_name, dv_T1_name

results_path = analysis_path + "Results/Performance_all/"

np.random.seed(global_seed)
datetime = start_string
T1_baseline = "DV_T1__25_Jan_2023__12.32.csv"
# ach + Track
NS_baseline = "NS_base__15_Apr_2023__14.31_track_and_pp.csv"
# all
# NS_baseline = "NS_base__15_Apr_2023__14.45_test.csv"

rf_results = pd.read_csv(results_path + "RF_" + datetime + ".csv", index_col="Unnamed: 0")
enet_results = pd.read_csv(results_path + "ENet_" + datetime + ".csv", index_col="Unnamed: 0")
if fixed_dummy_r2 == False:
    dum_results = pd.read_csv(results_path + "Dummy_" + datetime + ".csv", index_col="Unnamed: 0")
if fixed_dummy_r2 == True:
    dum_results = pd.read_csv(results_path + "Dummy_" + datetime + "_fixed.csv", index_col="Unnamed: 0")
ns_base_results = pd.read_csv(results_path + NS_baseline, index_col="Unnamed: 0")


rf_results['Model'] = "Random Forest"
enet_results['Model'] = "Elastic Net"
dum_results['Model'] = mean_baseline_name
# ns_name = 'Prior Ach., Grades, Track, IQ, Demo.'
ns_name = "Prior Ach. + School Track"
ns_base_results['Model'] = ns_name

if Additional_pred_checks == True:
    xticks = ["6 --> 8", "6 --> 9", "7 --> 9"]
if Additional_pred_checks == False:
    # xticks = ["5 --> 6", "5 --> 7", "5 --> 8", "5 --> 9"]
    # xticks = ["5 --> 6 (+1)", "5 --> 7 (+2)", "5 --> 8 (+3)", "5 --> 9 (+4)"]
    xticks = ["5-->6 (+1)", "5-->7 (+2)", "5-->8 (+3)", "5-->9 (+4)"]

if Compare_DV_T1 == True:
    # compare to sges_T1 only when sges is not dropped (for now)
    dv_t1_results = pd.read_csv(results_path + "DV_T1_{}.csv".format(start_string), index_col="Unnamed: 0")
    dv_t1_results['Model'] = dv_T1_name

    r2_res2 = pd.concat([rf_results[["index", "r2", "Model"]],
                         enet_results[["index", "r2", "Model"]],
                         dv_t1_results[["index", "r2", "Model"]],
                         # dum_results[["index", "r2", "Model"]],
                         ns_base_results[["index", "r2", "Model"]]], axis=0
                        )
    MAE_res2 = pd.concat([rf_results[["index", "MAE", "Model"]],
                          enet_results[["index", "MAE", "Model"]],
                          dv_t1_results[["index", "MAE", "Model"]],
                          # dum_results[["index", "MAE", "Model"]],
                          ns_base_results[["index", "r2", "Model"]]], axis=0
                         )
    save_name = "New_R2_DVT1" + datetime
    save_path = results_path + "Plots/"

    plot_results(x="index", y="r2", data=r2_res2, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="School Year", ylab="Prediction R Squared",
                 title="Comparison of Predictions", legend_pos="upper left",
                 xaxis_labs=xticks, y_lim=(0, 1),
                 order=[dv_T1_name, ns_name,
                        "Elastic Net", "Random Forest"]
                 )

    save_name = "New_MAE_DVT1" + datetime
    plot_results(x="index", y="MAE", data=MAE_res2, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="School Year", ylab="MAE",
                 title="Comparison of Predictions", legend_pos="upper left",
                 xaxis_labs=xticks, y_lim=(0,90),
                 order=[dv_T1_name, ns_name,
                        "Elastic Net", "Random Forest"]
                 )

else:
    r2_res = pd.concat([rf_results[["index", "r2", "Model"]],
                       enet_results[["index", "r2", "Model"]]], axis=0
                       )

    MAE_res = pd.concat([rf_results[["index", "MAE", "Model"]],
                       enet_results[["index", "MAE", "Model"]],
                       dum_results[["index", "MAE", "Model"]]], axis=0
                       )

    save_path = analysis_path + "Results/Performance_all/Plots/"

    save_name = "R2" + datetime

    plot_results(x="index", y="r2", data=r2_res, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="School Year", ylab=" Prediction R Squared",
                 title="Comparison of Predictions",
                 xaxis_labs=xticks)

    save_name = "MAE" + datetime
    plot_results(x="index", y="MAE", data=MAE_res, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="School Year", ylab="MAE",
                 title="Comparison of Predictions",
                 xaxis_labs=xticks)

print('done!')