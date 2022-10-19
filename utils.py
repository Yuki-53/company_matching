from sklearn.model_selection import train_test_split
from fuzzywuzzy import fuzz
# from cdifflib import CSequenceMatcher
import tensorflow as tf


def data_test_split(data):
    _, database = train_test_split(
        data, test_size = 0.2,
        stratify=data['is_duplicate'], random_state=53)
    _, query = train_test_split(
        database, test_size = 0.5,
        stratify=database['is_duplicate'], random_state=53)
    query = query[query['is_duplicate']==1]['name_1'].values.tolist()
    return database.reset_index(), query

# def standardize_text(df, text_field):
#     df[text_field] = df[text_field].str.lower()
#     df[text_field] = df[text_field].str.replace(r"[0-9().,!?@\'\`\"\_\n-_]", " ")
#     return df

# def seq_match(str1, str2):
#     return CSequenceMatcher(None, str1, str2).ratio()

def prep_data(data, company_name):
    data['name_1'] = company_name.lower().replace(r"[0-9().,!?@\'\`\"\_\n-_]", " ")
    data['len_str_1'] =  len(company_name)
    data['len_words_1'] =  len(company_name.split())
    data['name_2'] = data['name_2'].str.lower().replace(r"[0-9().,!?@\'\`\"\_\n-_]", " ")
    data['len_str_2'] =  data['name_2'].str.len()
    data['len_words_2'] =  data['name_2'].str.split().str.len()
    data['len_intersection'] = data.apply(lambda x: len(set(x['name_1'].split(' ')) & set(x['name_2'].split(' '))), axis = 1)
    data['fuzz_ratio'] = data.apply(lambda x: fuzz.ratio(x['name_1'], x['name_2']), axis = 1)
    data['jaccard'] = data.apply(lambda x: x['len_intersection']/(x['len_str_1'] + x['len_str_2'] - x['len_intersection']), axis=1)
    # data['seq_match'] = data.apply(lambda x: seq_match(x['name_1'], x['name_2']), axis=1)
    return data

def get_metrics(preds, true, ks = [1, 3, 5, 10, 20, 100, 1000]):
    for k in ks:
        acc_k = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
        acc_k.update_state([true], [preds])
        print('k:', acc_k.result().numpy())
