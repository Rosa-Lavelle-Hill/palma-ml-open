import pandas as pd

cat_vars = ["houschn", "houspan", "sext", "sctyp", "sex", "rueckp", "vorp",
            "switch_track", "borns", "bornf", "bornm", "langu"]
# todo: check par

# Also don't need dummies for aggreagte as just choosing most common/ taking the mean:
cat_vars_agg = ["houschn", "houspan", "sext", "houschn_C", "houspan_C", "sext_C", "sctyp", "sctyp_C"]

agg_mean = ["rueckp", "sex", "vorp"]
agg_mode = ["houschn", "houspan", "sext", "sctyp"]

drop_list = ["Unnamed: 0", "vpnr", "index", "laeng12345", "vpnrg_u", "School_Class", "clnr",
             "Unnamed: 0.1", "Unnamed: 0_C", "Unnamed: 0.1_C", "School_Class", "gbi_jz"
             ]

teacher_cols = pd.read_csv("Data/MetaData/Constructs/Teacher/Teacher_cols_all_6.csv", index_col=[0])

# block_file = "Data/MetaData/Variable_blocks/variable_blocks_final_with_Teacher_survey_vs_not.csv"
block_file = "Data/MetaData/Variable_blocks/variable_blocks_final_with_Teacher_change_cog_and_track.csv"
# changes = "survey_vs_not"
changes = "track_changes"

parental_degree_expectations = ["parental_expectation_degree", "aspns", "aspds", "aspnp", "aspdp"]

# run options
include_T5 = False
add_teacher_vars = True
Compare_DV_T1 = True

# imputation
imputer_model = "HH"
imputer_max_iter = 100

# drop decisions
drop_p_degree_expect=True
drop_houspan=False

# baseline names
mean_baseline_name = "Mean Prediction Baseline"
dv_T1_name = "Prior Achievement Baseline"