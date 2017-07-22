import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
import process_data as proc
import heapq
from sets import Set
from sklearn.metrics import roc_curve
import roc
import plot_confusion_matrix as pcm

def adaboost(X_train, X_test, y_train, y_test, grid_search = None):
    '''
    This function trains a boosted tree classifier (adaboost) with the training data and
    returns the predicted values for the "test" data (validation data)
    Input: traning and test data,
    grid_search: default to none, if small or large run do_grid_search and fit with best paramaters.
    '''
    if grid_search == 'small':
        #best_score, best_params = do_grid_search(X_train, X_test, y_train, y_test, search_type = 'small')
        # Initalize our model here
        # original est = RandomForestClassifier()
        est = AdaBoostClassifier(base_estimator=None, algorithm='SAMME.R', random_state=42)
        param_grid = {"n_estimators": [5,10,50,100,300], "learning_rate": [0.1,0.5,1.0,1.2]}
        # Plug in our model, params dict, and the number of jobs, then .fit()
        gs_cv = GridSearchCV(est, param_grid, n_jobs=3).fit(X_train, y_train)
        # return the best score and the best params
        print(gs_cv.best_score_, gs_cv.best_params_)
        best_params = gs_cv.best_params_
        adaboost = AdaBoostClassifier(base_estimator=None, algorithm='SAMME.R', random_state=42, n_estimators = best_params['n_estimators'], learning_rate = best_params['learning_rate'])
        adaboost.fit(X_train, y_train)
        y_predict = adaboost.predict(X_test)
        return adaboost, y_predict, best_params

    else:
        adaboost = AdaBoostClassifier(base_estimator=None, algorithm='SAMME.R', random_state=42, n_estimators = 50, learning_rate = 0.1)
        adaboost.fit(X_train, y_train)
        y_predict = adaboost.predict(X_test)
        return adaboost, y_predict, best_params

def randomforest(X_train, X_test, y_train, y_test, grid_search = None):
    '''
    This function trains a random forest classifier with the training data and
    returns the predicted values for the "test" data (validation data)
    Input: traning and test data,
    grid_search: default to none, if small or large run do_grid_search and fit with best paramaters.
        best_grid_params for original training data
            {'max_depth': 5,
            'max_features': 30,
            'min_samples_leaf': 2,
            'min_samples_split': 10}
    '''
    if grid_search == 'small':
        best_score, best_params = do_grid_search(X_train, X_test, y_train, y_test, search_type = 'small')
        rf = RandomForestClassifier(class_weight = 'balanced_subsample', n_estimators = 50, max_depth = best_params['max_depth'], min_samples_leaf= best_params['min_samples_leaf'], min_samples_split = best_params['min_samples_split'])
        rf.fit(X_train, y_train)
        y_predict = rf.predict(X_test)
        return rf, y_predict, best_params
    elif grid_search =='large':
        best_score, best_params = do_grid_search(X_train, X_test, y_train, y_test, search_type = 'large')
        rf = RandomForestClassifier(class_weight = 'balanced_subsample', n_estimators = best_params['n_estimators'], max_depth = best_params['max_depth'], min_samples_leaf= best_params['min_samples_leaf'], min_samples_split = 10)
        rf.fit(X_train, y_train)
        y_predict = rf.predict(X_test)
        return rf, y_predict, best_params
    else:
        rf = RandomForestClassifier(class_weight = 'balanced_subsample', n_estimators = 50, max_depth = 5, min_samples_leaf= 2, min_samples_split = 10)
        rf.fit(X_train, y_train)
        y_predict = rf.predict(X_test)
        return rf, y_predict

def do_grid_search(X_train, X_test, y_train, y_test, search_type = 'small'):
    '''
    Run a small or large grid search and return the best score and best paramaters
    Input: Train and test data,
          search_type: default none, small or large
    Output: best_score, best params

    '''
    # Initalize our model here
    # original est = RandomForestClassifier()
    est = RandomForestClassifier(bootstrap=True, criterion = "gini", class_weight = "balanced_subsample")

    # Here are the params we are tuning, ie,
    # if you look in the docs, all of these are 'nobs' within the GradientBoostingClassifier algo.
    if search_type == 'small':
        param_grid = {"n_estimators": [10,20,50,100],"max_depth": [3, 5, 10, 30, 50, 100], "max_depth":[2,5,10,30],"max_features": [1, 3, 10, 30],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [2, 3, 10]}

    elif search_type == 'large':
        param_grid = {"n_estimators": [10,20,50,100],"max_depth": [3, 5, 10, 30, 50, 100], "max_depth":[2,5,10,30],"max_features": [1, 3, 10, 30],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [2, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "class_weight":[None, "balanced_subsample"]}

    # Plug in our model, params dict, and the number of jobs, then .fit()
    gs_cv = GridSearchCV(est, param_grid, n_jobs=3).fit(X_train, y_train)
    # return the best score and the best params
    return gs_cv.best_score_, gs_cv.best_params_

def set_threshold_recall(model, X_train, X_test, y_train, y_test):
    '''
    This function is only useful when the HTS screen has been scored.
    In that case it can be used to set a threshold and review the preformance of the model.
    Set a threshold for (median of) maximal recall.
        The function iterates through 100 possible threshold values (0 to 1.0) and records the recall and precision for each threshold. Then it finds the unique values in the recall list and selects the index values for the recall value closest to 1.0 (but not 1.0). The median threshold from the list that would return the same recall is returned along with the stats for that threshold.
    Input
        model -  trained on fragment library.
        X - features
        y -scored hits
    Output
        precision - at selected threshold
        recall- at selected threshold
        median_recall - list of index for median recall (same as index for corresponding threshold), 1 value if odd number of values in best_recall_index or 2 if even
        threshold[median_recall[0]] - threshold cooresponding to the median recall
    '''
    precision = []
    recall = []
    f1score = []
    threshold = list(np.linspace(0.0,1.0,100))
    for i in threshold:
    	pprob = model.predict_proba(X_test)
    	pdf = pd.DataFrame(pprob)
    	pdf['myH'] = pdf[1].map(lambda x: 1 if x>i else 0)
    	my_pred = pdf['myH'].values
    	precision.append(precision_score(y_test, my_pred))
    	recall.append(recall_score(y_test, my_pred))
    	f1score.append(f1_score(y_test, my_pred))
    recall_set = list(Set(recall))
    recall_set.sort()
    '''
    best_recall= recall_set[-2]
    best_recall_index = [i for i,v in enumerate(recall) if v == best_recall]
    median_recall = [best_recall_index[i] for i in range((len(best_recall_index)/2) - (1 if float(len(best_recall_index)) % 2 == 0 else 0), len(best_recall_index)/2+1)]
    return precision, recall, median_recall, threshold[median_recall[0]]
    '''
    #replace last section with below if want to use second best recall
    #this was done because the best recall did not yeild usefull results (too many FP)
    secondbest_recall= recall_set[-3]
    best_recall_index = [i for i,v in enumerate(recall) if v == secondbest_recall]
    bestprecision_recall = best_recall_index[-1]
    return precision, recall, bestprecision_recall, threshold[bestprecision_recall]





def set_threshold_precision(model, X_train, X_test, y_train, y_test):
    '''
    This function is only useful when the HTS screen has been scored.
    In that case it can be used to set a threshold and review the preformance of the model.
    Set a threshold for (median of) maximal precision.
        The function iterates through 100 possible threshold values (0 to 1.0) and records the recall and precision for each threshold. Then it finds the unique values in the recall list and selects the index values for the recall value closest to 1.0 (but not 1.0). The median threshold from the list that would return the same precision is returned along with the stats for that threshold.
    Input
        model -  trained on fragment library.
        X - features
        y -scored hits
    Output
        precision - at selected threshold
        recall- at selected threshold
        median_precision - list of index for median precision (same as index for corresponding threshold), 1 value if odd number of values in best_precision_index or 2 if even
        threshold[median_precision[0]] - threshold cooresponding to the median precision
    '''
    precision = []
    recall = []
    f1score = []
    threshold = list(np.linspace(0.0,1.0,100))
    for i in threshold:
    	pprob = model.predict_proba(X_test)
    	pdf = pd.DataFrame(pprob)
    	pdf['myH'] = pdf[1].map(lambda x: 1 if x>i else 0)
    	my_pred = pdf['myH'].values
    	precision.append(precision_score(y_test, my_pred))
    	recall.append(recall_score(y_test, my_pred))
    	f1score.append(f1_score(y_test, my_pred))
    precision_set = list(Set(precision))
    precision_set.sort()
    sugg_precision= precision_set[-4]
    sugg_precision_index = [i for i,v in enumerate(precision) if v == sugg_precision]
    median_precision = [sugg_precision_index[i] for i in range((len(sugg_precision_index)/2) - (1 if float(len(sugg_precision_index)) % 2 == 0 else 0), len(sugg_precision_index)/2+1)]
    bestrecall_precision = sugg_precision_index[-1]
    return precision, recall, median_precision, threshold[median_precision[0]]

def print_threshold(model, X_train, X_test, y_train, y_test, threshold):
    '''
    Get stats for a trained model at a given threshold.
    Input
        model - trained with fragment (or equivilant) data
        X - features
        y - scored hits
        threshold - value above which a given probability is scored a hit
    Output
        precision
        recall
        fpr - false positiv Rate
        fpr_test - list of fpr for given thresholds
        tpr_test - list of tpr for given thresholds
        cm - confusion matrix
    '''
    pprob = model.predict_proba(X_test)
    pdf = pd.DataFrame(pprob)
    #print(pdf)
    pdf['myH'] = pdf[1].map(lambda x: 1 if x>threshold else 0)
    my_pred = pdf['myH'].values
    precision =  precision_score(y_test, my_pred)
    recall = recall_score(y_test, my_pred)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pprob[:,1])
    #print(np.array([['TN','FP'],['FN', 'TP']]))
    cm = confusion_matrix(y_test, my_pred)
    fpr = cm[0][1]/float(cm[0][1] + cm[0][0])
    return precision, recall, fpr, fpr_test, tpr_test, cm


def feature_importance(features, rf):
    feature_col = features.columns
    importances = rf.feature_importances_
    # print "\nFeature and importances unsorted"
    # for n, (f,i) in enumerate(zip(features, importances)):
    #     print "{0}\t{1}\t{2:0.3f}".format(n, f, i)
    fi_idx = np.argsort(rf.feature_importances_)[::-1]
    print("\n top five features by importance:", list(features.columns[fi_idx[:5]]))
    print("\n top five importance scores :", list(importances[fi_idx[:5]]))

def feature_description(feature_list):
    '''
    Develop a table with phisochemical features in top ten of feature importance.
    Not extensive yet, needs to be finished.
    Input: feature_list - list of features to be described
    Output: pandas dataframe with descriptions in same order
    '''
    mqn = pd.read_csv('MQN_descriptors.csv')
    descrip_list= []
    for i in feature_list:
        if i[:3]=='MQN':
            descrip_list.append('number of ' + mqn.iloc[int(i[3:]),1])
        elif i[:3]=='Bit':
            descrip_list.append('MorganFingerPrint')
        elif i[:7]=='smr_VSA':
            descrip_list.append('MOE-type descriptors using MR contributions and surface area contributions')
        elif i[:9]=='slogp_VSA':
            descrip_list.append('MOE-type descriptors using LogP contributions and surface area contributions')
        elif i[:8]=='peoe_VSA':
            descrip_list.append('MOE-type descriptors using partial charges and surface area contributions')
        elif i[:4]=='TPSA':
            descrip_list.append('Topological PSA (Polar Surface Area)')
        elif i[:5]=='SlogP':
            descrip_list.append('Log of the octanol/water partition coefficient')
        elif i[:3]=='SMR':
            descrip_list.append('Molecular refractivity')
        else:
            descrip_list.append('see RDkit documentation')
    #print(len(descrip_list))
    #print(len(feature_list))
    feature_descrip = pd.DataFrame(np.column_stack([feature_list, descrip_list]),
                               columns=['Feature', 'Description'])
    #feature_descrip = pd.DataFrame({'description': descrip_list,'feature': feature_list})
    return feature_descrip







def plot_features(features, model, name, n=10):
    '''
    Plot n number of most important features for a trained model
    Input
        features - feature space, must be the same between training and HTS data, only used to get column names
        model - trained on fragment (or equivilant) data
        name - string to save under
        n - number of features to display
    '''
    feature_names = features.columns
    importances = model.feature_importances_
    fi_idx = np.argsort(model.feature_importances_)[::-1]
    topn_idx = fi_idx[:n]
    #std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    plt.figure()
    plt.title("Feature Importance")
    plt.bar(range(10), importances[topn_idx], color="r", align="center")
    plt.xticks(range(10), feature_names[topn_idx], rotation=90)
    plt.xlim([-1, 10])
    plt.tight_layout()
    #plt.ylim([0.01, 0.1])
    plt.figure(figsize = (7,7))
    plt.savefig('static/Feature_Importance_{}.png'.format(name), dpi = 900)
    plt.close()
    f_descrip = feature_description(feature_names[topn_idx])
    return f_descrip


if __name__ == '__main__':
    df = proc.read_data()
    #df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # use all features and yfill (no NaNs, filled with 0)
    features, yfill = proc.features_yfill(df)
    #train test split at 20%
    X_train, X_test, y_train, y_test = train_test_split(features, yfill, test_size=0.20, random_state=1, stratify =yfill)

    #Optional: oversampling of minority class for training purposes
    #X_train_over, y_train_over = proc.oversample(X_train,y_train, r = 0.3)
    #rffit, y_predict = rf.randomforest(X_train_over, X_test, y_train_over, y_test, num_est=50, cls_w = 'balanced_subsample')

    #fit the Random Forest classifier: would like to add in a grid search
    rffit, y_predict = randomforest(X_train.values, X_test, y_train.values, y_test)

    # Use below to run a grid search .... takes to long to work right now
    #rffit, y_predict = rf.randomforest(X_train.values, X_test, y_train.values, y_test, grid_search = 'small')

    #pickle the fit model for use with test data
    #proc._pickle(rffit, 'RFC_fit.pkl')

    #set_threshold_recall is a function which determines the threshold to set such that recall is optimized (the median of the available thresholds that return the second best recall (not 1.0))
    precision_list, recall_list, median_recall_index, medianrecall_threshold = set_threshold_recall(rffit, X_train, X_test, y_train, y_test)

    #print_threshold uses the trained model and the selected threshold (in this case recall optimized) to return listed statistics
    precision, recall, fpr, fpr_test, tpr_test, cm = print_threshold(rffit, X_train, X_test, y_train, y_test, medianrecall_threshold)
    r_cm = pd.DataFrame(cm)
    #proc._pickle(medianrecall_threshold, 'medianrecall_threshold.pkl')

    #make a pd.dataframe of the stats for display
    recall_opt_stats = pd.DataFrame([[format(medianrecall_threshold, '.2f'),format(recall, '.2f'), format(fpr, '.2f'), format(precision, '.2f'), ]], columns = ['Suggested Threshold','True Positive Rate (Recall)', 'False Positive Rate (Fall-out)','Precision'])

    # repeat the threshold selection process for precision optimization
    p_precision, p_recall, p_median_precision, threshold_precision = set_threshold_precision(rffit, X_train, X_test, y_train, y_test)
    p_precision, p_recall, p_fpr, p_fpr_test, p_tpr_test, p_cm = print_threshold(rffit, X_train, X_test, y_train, y_test, threshold_precision)
    pcm.plot_confusion_matrix_basic(p_cm, classes = ['Not a Hit', 'Hit'], name = 'precision_CM')
    p_cm = pd.DataFrame(p_cm)
    precision_opt_stats = pd.DataFrame([[format(threshold_precision, '.2f'),format(p_recall, '.2f'), format(p_fpr, '.2f'), format(p_precision, '.2f'), ]], columns = ['Suggested Threshold','True Positive Rate (Recall)', 'False Positive Rate (Fall-out)','Precision'])
