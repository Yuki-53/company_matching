import pandas as pd
import utils
import numpy as np

from catboost import CatBoostClassifier
from tqdm import tqdm

def get_pred(database, company_name, model, feats):
    request = utils.prep_data(database[['name_2']], company_name)
    pred_proba = model.predict_proba(request[feats])[:, 1]
    return pred_proba


def main():
    data = pd.read_csv('data/train.csv')
    database, query = utils.data_test_split(data)
    
    model = CatBoostClassifier() 
    model.load_model('catboost_model.bin')
    feat_cols = [
        'name_1', 'name_2', 'len_intersection', 'len_str_1', 'len_words_1',
        'len_str_2', 'len_words_2','fuzz_ratio', 'jaccard']
    
    pred_probas = np.zeros(shape=(len(query), len(database)))
    true_vals = np.zeros(shape=(len(query), 1))
    for company_id in tqdm(range(len(query))):
        company_name = query[company_id]
        true_val = database[database['name_1'] == company_name].index[0]
        pred_proba = get_pred(database, company_name, model, feat_cols)
        true_vals[company_id] = true_val
        pred_probas[company_id] = pred_proba
    labels = range(len(database))
    utils.get_metrics(pred_probas, true_vals, labels)

if __name__ == "__main__":
    main()
