import pandas as pd
import numpy as np
from Functions.Plotting import plot_results
from RUN_Predict_same_year import start_string, global_seed, analysis_path, xticks
from Fixed_params import Compare_DV_T1

fixed_dummy_r2 = False

results_path = analysis_path + "Results/Performance_all/"

np.random.seed(global_seed)
datetime = start_string

rf_results = pd.read_csv(results_path + "RF_" + datetime + ".csv", index_col="Unnamed: 0")
enet_results = pd.read_csv(results_path + "ENet_" + datetime + ".csv", index_col="Unnamed: 0")
if fixed_dummy_r2 == False:
    dum_results = pd.read_csv(results_path + "Dummy_" + datetime + ".csv", index_col="Unnamed: 0")
if fixed_dummy_r2 == True:
    dum_results = pd.read_csv(results_path + "Dummy_" + datetime + "_fixed.csv", index_col="Unnamed: 0")

rf_results['Model'] = "Random Forest"
enet_results['Model'] = "Elastic Net"
dum_results['Model'] = "Mean (train) Maths Ability Grade 5 Baseline"

if Compare_DV_T1 == True:
    # compare to sges_T1 only when sges is not dropped (for now)
    dv_t1_results = pd.read_csv(results_path + "DV_T1_{}.csv".format(start_string), index_col="Unnamed: 0")
    dv_t1_results['Model'] = "Maths Ability Grade 5 Baseline"

    r2_res2 = pd.concat([rf_results[["index", "r2", "Model"]],
                         enet_results[["index", "r2", "Model"]],
                         dv_t1_results[["index", "r2", "Model"]],
                         dum_results[["index", "r2", "Model"]]], axis=0
                        )
    MAE_res2 = pd.concat([rf_results[["index", "MAE", "Model"]],
                          enet_results[["index", "MAE", "Model"]],
                          dv_t1_results[["index", "MAE", "Model"]],
                          dum_results[["index", "MAE", "Model"]]], axis=0
                         )
    save_name = "R2_DVT1" + datetime
    save_path = results_path + "Plots/"

    plot_results(x="index", y="r2", data=r2_res2, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="Schooling Year (Grade)", ylab="Prediction R Squared",
                 title="Comparison of Predictions", legend_pos="upper left",
                 xaxis_labs=xticks, y_lim=(0, 1))

    save_name = "MAE_DVT1" + datetime
    plot_results(x="index", y="MAE", data=MAE_res2, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="Schooling Year (Grade)", ylab="MAE",
                 title="Comparison of Predictions", legend_pos="upper left",
                 xaxis_labs=xticks)

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
                 xlab="Schooling Year (Grade)", ylab="R Squared",
                 title="Comparison of Predictions",
                 xaxis_labs=xticks)

    save_name = "MAE" + datetime
    plot_results(x="index", y="MAE", data=MAE_res, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="Schooling Year (Grade)", ylab="MAE",
                 title="Comparison of Predictions",
                 xaxis_labs=xticks)

print('done!')