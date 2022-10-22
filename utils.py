from sklearn.model_selection import train_test_split
from fuzzywuzzy import fuzz
from sklearn.metrics import top_k_accuracy_score


def data_test_split(data):
    _, database = train_test_split(
        data, test_size = 0.2,
        stratify=data['is_duplicate'], random_state=53
    )
    _, query = train_test_split(
        database, test_size = 0.04,
        stratify=database['is_duplicate'], random_state=53
    )
    query = query[query['is_duplicate']==1]['name_1'].values.tolist()
    return database.reset_index(), query


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
    return data


def get_metrics(preds, true, labels, ks = [5, 10, 20, 50]):
    print('database size - ', len(preds[0]))
    print('test size - ', len(preds))
    for k in ks:
        print(f'top {k} accuracy score:', top_k_accuracy_score(true, preds, k=k, labels=labels))
