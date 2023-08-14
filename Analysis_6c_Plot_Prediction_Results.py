import pandas as pd
import numpy as np
from Functions.Plotting import plot_results
from RUN_Analysis_6 import start_string, global_seed, analysis_path, Choose_drop_DV_T1

results_path = analysis_path + "Results/Performance_all/"
save_path = analysis_path + "Results/Performance_all/Plots/"

np.random.seed(global_seed)
datetime = start_string

rf_results = pd.read_csv(results_path + "RF_" + datetime + ".csv", index_col=[0])
enet_results = pd.read_csv(results_path + "ENet_" + datetime + ".csv", index_col=[0])
dum_results = pd.read_csv(results_path + "Dummy_" + datetime + ".csv", index_col=[0])

if Choose_drop_DV_T1 == False:
    dv_t1_results = pd.read_csv(results_path + "DV_T1_" + datetime + ".csv", index_col="Unnamed: 0")
    dv_t1_results['Model'] = "DV Time 1 Baseline"
    r2_res2 = pd.concat([rf_results[["index", "r2", "Model"]],
                         enet_results[["index", "r2", "Model"]],
                         dv_t1_results[["index", "r2", "Model"]]], axis=0
                        )
    MAE_res2 = pd.concat([rf_results[["index", "MAE", "Model"]],
                          enet_results[["index", "MAE", "Model"]],
                          dum_results[["index", "MAE", "Model"]],
                          dv_t1_results[["index", "MAE", "Model"]]], axis=0
                         )
    save_name = "R2_DVT1" + datetime
    plot_results(x="index", y="r2", data=r2_res2, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="Schooling Year (Grade)", ylab="R Squared",
                 title="Comparison of Predictions", legend_pos="lower right",
                 xaxis_labs=['5-->6', '6-->7', '7-->8', '8-->9', '9-->10'])
    save_name = "MAE_DVT1" + datetime
    plot_results(x="index", y="MAE", data=MAE_res2, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="Schooling Year (Grade)", ylab="MAE",
                 title="Comparison of Predictions", legend_pos="lower right",
                 xaxis_labs=['5-->6', '6-->7', '7-->8', '8-->9', '9-->10'])


rf_results['Model'] = "Random Forest"
enet_results['Model'] = "Elastic Net"
dum_results['Model'] = "Mean Baseline"

r2_res = pd.concat([rf_results[["index", "r2", "Model"]],
                   enet_results[["index", "r2", "Model"]]], axis=0
                   )

MAE_res = pd.concat([rf_results[["index", "MAE", "Model"]],
                   enet_results[["index", "MAE", "Model"]],
                   dum_results[["index", "MAE", "Model"]]], axis=0
                   )

save_name = "R2" + datetime
plot_results(x="index", y="r2", data=r2_res, colour='Model',
             save_path=save_path, save_name=save_name,
             xlab="Schooling Year (Grade)", ylab="R Squared",
             title="Comparison of Predictions",
             xaxis_labs=['5-->6', '6-->7', '7-->8', '8-->9', '9-->10'])

save_name = "MAE" + datetime
plot_results(x="index", y="MAE", data=MAE_res, colour='Model',
             save_path=save_path, save_name=save_name,
             xlab="Schooling Year (Grade)", ylab="MAE",
             title="Comparison of Predictions",
             xaxis_labs=['5-->6', '6-->7', '7-->8', '8-->9', '9-->10'])

print('done!')