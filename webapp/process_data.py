from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import Counter
import pickle

def _pickle(obj, filename):
    # dumps an object into a pickle file
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def _unpickle(filename):
    #unpickles an object
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def features_y_user(data, hit_col, start_col, end_col):
    '''
    Reads in dataframe and column names for the scored hits, feature start column
    and feature end_column.
    '''
    y = data.loc[:,hit_col]
    yfill = y.fillna(0)
    features = data.loc[:,start_col:end_col]
    #featuresfill = features.fillna(0)
    return features, yfill


def features_HTS_user(data, start_col, end_col, id_col):
    '''
    Reads in dataframe and column names for the molecule ids, feature start column
    and feature end_column.
    '''
    ids = data.loc[:,id_col]
    features = data.loc[:,start_col:end_col]
    return features, ids

def id_scores(data, id_col, hit_col):
    '''
    Reads in dataframe and column names for the molecule ids, and scored hits columns.
    '''
    ids = data.loc[:,id_col]
    hits = data.loc[:,hit_col]
    return hits, ids

def oversample(X, y, r = 0.5):
    #example at http://contrib.scikit-learn.org/imbalanced-learn/generated/imblearn.over_sampling.RandomOverSampler.html
    print('Original dataset shape {}'.format(Counter(y)))
    ros = RandomOverSampler(ratio = r, random_state=42)
    X_res, y_res = ros.fit_sample(X, y)
    print('Resampled dataset shape {}'.format(Counter(y_res)))
    return X_res, y_res

if __name__ == '__main__':
    data = read_data()
    features, yfill = features_yfill(data)
    X_train, X_test, y_train, y_test = train_test_split(features, yfill, test_size=0.20, random_state=42, stratify =yfill)
    rng_seed = 2 # set random number generator seed
    np.random.seed(rng_seed)
    X_train_over, y_train_over = oversample(X_train,y_train, r = 0.3)
