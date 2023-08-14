import re
import os
import pandas as pd
from Functions.Aggregate_Features import Class_level
from RUN_Analysis_5 import version


data_path = "Data/Initial_Preprocess/"
directory_bits = os.fsencode("Data/Initial_Preprocess/")
save_path = "Data/Initial_Preprocess/Data_with_Aggregate_Features/"

# load aggregate meta data
aggregate_meta_data = pd.read_csv("Data/MetaData/Aggregate_Cols.csv", index_col=[0])

if __name__ == "__main__":
    print('Calculating class aggregate level features for each wave')
    Class_level(directory_bits=directory_bits,
                data_path=data_path,
                aggregate_meta_data=aggregate_meta_data,
                save_path=save_path, version=version)

    print('done!')