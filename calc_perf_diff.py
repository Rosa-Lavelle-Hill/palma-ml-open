import pandas as pd

# A1 - next year prediction
dvt1_res = pd.read_csv("Analysis_2_with_sges_T1/Results/Performance_all/DV_T1__25_Jan_2023__12.15.csv")
ns_res = pd.read_csv("Analysis_2_with_sges_T1/Results/Performance_all/NS_base__15_Apr_2023__14.18_just_track_and_pp.csv")
enet_res = pd.read_csv("Analysis_2_with_sges_T1/Results/Performance_all/ENet__25_Jan_2023__12.15.csv")
rf_res = pd.read_csv("Analysis_2_with_sges_T1/Results/Performance_all/RF__25_Jan_2023__12.15.csv")

dvt1_res = dvt1_res.sort_values(by='index', axis=0, ascending=True).reset_index()
ns_res = ns_res.sort_values(by='index', axis=0, ascending=True).reset_index()
enet_res = enet_res.sort_values(by='index', axis=0, ascending=True).reset_index()
rf_res = rf_res.sort_values(by='index', axis=0, ascending=True).reset_index()

ml_av = (enet_res + rf_res)/2
ml_best = pd.concat([enet_res, rf_res]).groupby(level=0).max()
dvt1_base_av_diff = round(ml_av - dvt1_res, 2)
ns_base_av_diff = round(ml_av - ns_res, 2)
dvt1_base_best_diff = round(ml_best - dvt1_res, 2)
ns_base_best_diff = round(ml_best - ns_res, 2)

ml_av = round(ml_av, 2)
ml_best = round(ml_best, 2)
enet_res = round(enet_res, 2)
rf_res = round(rf_res, 2)
ns_res = round(ns_res, 2)
dvt1_res = round(dvt1_res, 2)

save_path = "Results_general/Analysis_2/"

dvt1_res.to_csv(save_path+"dvt1_res.csv")
ns_res.to_csv(save_path+"ns_res.csv")
enet_res.to_csv(save_path+"enet_res.csv")
rf_res.to_csv(save_path+"rf_res.csv")
ml_av.to_csv(save_path+"ml_average.csv")
ml_best.to_csv(save_path+"ml_best.csv")
dvt1_base_av_diff.to_csv(save_path+"dvt1_base_av_diff.csv")
ns_base_av_diff.to_csv(save_path+"ns_base_av_diff.csv")
dvt1_base_best_diff.to_csv(save_path+"dvt1_base_best_diff.csv")
ns_base_best_diff.to_csv(save_path+"ns_base_best_diff.csv")

# A2 - varying future prediction
dvt1_res = pd.read_csv("Predict_X_years_ahead_with_sges_T1/Results/Performance_all/DV_T1__25_Jan_2023__12.32.csv")
ns_res = pd.read_csv("Predict_X_years_ahead_with_sges_T1/Results/Performance_all/NS_base__15_Apr_2023__14.31_track_and_pp.csv")
enet_res = pd.read_csv("Predict_X_years_ahead_with_sges_T1/Results/Performance_all/ENet__25_Jan_2023__12.32.csv")
rf_res = pd.read_csv("Predict_X_years_ahead_with_sges_T1/Results/Performance_all/RF__25_Jan_2023__12.32.csv")

dvt1_res = dvt1_res.sort_values(by='index', axis=0, ascending=True).reset_index()
ns_res = ns_res.sort_values(by='index', axis=0, ascending=True).reset_index()
enet_res = enet_res.sort_values(by='index', axis=0, ascending=True).reset_index()
rf_res = rf_res.sort_values(by='index', axis=0, ascending=True).reset_index()

ml_av = (enet_res + rf_res)/2
ml_best = pd.concat([enet_res, rf_res]).groupby(level=0).max()
dvt1_base_av_diff = round(ml_av - dvt1_res, 2)
ns_base_av_diff = round(ml_av - ns_res, 2)
dvt1_base_best_diff = round(ml_best - dvt1_res, 2)
ns_base_best_diff = round(ml_best - ns_res, 2)

ml_av = round(ml_av, 2)
ml_best = round(ml_best, 2)
enet_res = round(enet_res, 2)
rf_res = round(rf_res, 2)
ns_res = round(ns_res, 2)
dvt1_res = round(dvt1_res, 2)

save_path = "Results_general/Predict_X_years_ahead/"

dvt1_res.to_csv(save_path+"dvt1_res.csv")
ns_res.to_csv(save_path+"ns_res.csv")
enet_res.to_csv(save_path+"enet_res.csv")
rf_res.to_csv(save_path+"rf_res.csv")
ml_av.to_csv(save_path+"ml_average.csv")
ml_best.to_csv(save_path+"ml_best.csv")
dvt1_base_av_diff.to_csv(save_path+"dvt1_base_av_diff.csv")
ns_base_av_diff.to_csv(save_path+"ns_base_av_diff.csv")
dvt1_base_best_diff.to_csv(save_path+"dvt1_base_best_diff.csv")
ns_base_best_diff.to_csv(save_path+"ns_base_best_diff.csv")

print("done")