import Credentials.DeepL as dl
from deep_translator import DeepL

def translator(list, source='de', target='en', api_key=dl.access_key):
    # use the translate_batch function
    translated = DeepL(api_key=api_key, source=source, target=target).translate_batch(list)
    return translated


# file_name = "../Data/All.sav"
#
# df_all, meta = pyreadstat.read_sav(file_name, encoding="latin1")
#
# # create a dictionary of column_names:column_labels
# german_eng = {}
#
# column_labels = dict(zip(meta.column_names, meta.column_labels))
#
# i=0
# stop=len(column_labels)+1
# for key, item in column_labels.items():
#     if i>stop:
#         break
#     if item != None:
#
#         # translate to german
#         # gs = goslate.Goslate()
#         # english_item = gs.translate(item, 'en')
#         time.sleep(2)
#
#         # Add to dict
#         english_item = english_item.replace(" ", "_")
#         german_eng[item] = english_item
#     i=i+1
#
# # to dataframe
# df = pd.DataFrame.from_dict(german_eng, orient='index')
# df.reset_index(inplace=True)
# df.columns = ["German", "English"]
#
# # save as .csv
# df.to_csv("Data/MetaData/EngTrans.csv")
#
# print('done!')
