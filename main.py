import pandas as pd
import utils
from catboost import CatBoostClassifier


if __name__ == "__main__":
    data = pd.read_csv('data/train.csv')
    database, query = utils.data_test_split(data)
    request = database[['name_2']].copy()
    company_id = int(input())
    company_name = query[company_id]
    true_val = [0]*len(database)
    ind = database[database['name_1'] == company_name].index[0]
    true_val[ind] += 1
    request = utils.prep_data(request, company_name)
    feat_cols = [
        'name_1', 'name_2', 'len_intersection', 'len_str_1', 'len_words_1',
        'len_str_2', 'len_words_2','fuzz_ratio', 'jaccard']
    model = CatBoostClassifier() 
    model.load_model('catboost_model.bin')
    pred_proba = model.predict_proba(request[feat_cols])[:, 1]
    utils.get_metrics(pred_proba, true_val)
