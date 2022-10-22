import pandas as pd
import utils
from catboost import CatBoostClassifier
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    data = pd.read_csv('data/train.csv')
    database, query = utils.data_test_split(data)
    request = database[['name_2']].copy()
    
    model = CatBoostClassifier() 
    model.load_model('catboost_model.bin')
    feat_cols = [
            'name_1', 'name_2', 'len_intersection', 'len_str_1', 'len_words_1',
            'len_str_2', 'len_words_2','fuzz_ratio', 'jaccard'
        ]
    
    pred_probas = np.zeros(shape=(len(query), len(database)))
    true_vals = np.zeros(shape=(len(query), 1))
    for company_id in tqdm(range(len(query))):
        company_name = query[company_id]
        true_val = database[database['name_1'] == company_name].index[0]
        request = utils.prep_data(request, company_name)
        pred_proba = model.predict_proba(request[feat_cols])[:, 1]
        true_vals[company_id] = true_val
        pred_probas[company_id] = pred_proba
    labels = range(len(database))
    utils.get_metrics(pred_probas, true_vals, labels)
