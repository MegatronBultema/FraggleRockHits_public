from flask import Flask, render_template, request, jsonify, redirect, url_for
from data import Articles
import pandas as pd
import numpy as np
import re
#from XGBoostModel import *
import RandomForest_webapp as rf
import HTS_analysis as hts_a
import process_data as proc
import roc
from sklearn.ensemble import RandomForestClassifier
from flask import send_from_directory
import plot_confusion_matrix as pcm
from sklearn.metrics import confusion_matrix
from flask_pymongo import PyMongo
from pymongo import MongoClient
import flask_login


import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS2 = set(['csv'])

app = Flask(__name__)
login_manager = flask_login.LoginManager()
login_manager.init_app(app)
mongo = PyMongo(app)
client = MongoClient()

'''
Code for login and registration has been removed to protect user privacy.
'''

class User(flask_login.UserMixin):
    pass

@login_manager.user_loader
def user_loader(email):
    pass

@login_manager.request_loader
def request_loader(request):
    pass


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':

        return '''
               <form action='login' method='POST'>
                <input type='text' name='email' id='email' placeholder='email'></input>
                <input type='password' name='pw' id='pw' placeholder='password'></input>
                <input type='submit' name='submit'></input>
               </form>
               '''

    email = request.form['email']
    if '''..... Removed for security'''
        return redirect(url_for('upload_file'))

    return 'Bad login'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS2

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/register', methods = ['GET', 'POST'])
def register_user():
    client = MongoClient()
    if request.method == 'POST':
        if request.form['submit'] == "Acedemic":
            pass
            '''Removed for security'''
        if request.form['submit'] == "Comercial":
            pass
            '''Removed for security'''
        return render_template('registration_completed.html')
    return render_template('registration.html')


@app.route('/fraggle', methods=['GET', 'POST'])
def upload_file():
    '''
    Function to run step 1) Fraggle. If method is GET will return the fraggle.html. If POST will check if the uploaded file is a .csv file and will save it in uploads. Collectes the sqecified column names for the users data and returns a redirect to the uploaded_file function below. redirect(url_for (.....)) will send the first variable into the specified function but then the rest go into the query string.
    '''
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            hit_col = request.form['hit']
            start_col = request.form['start']
            end_col = request.form['end']
            id_col = request.form['id']
            return redirect(url_for('uploaded_file', filename=filename, hit_col = hit_col, start_col = start_col, end_col = end_col, id_col=id_col))
    #Return frggle.html if no file has yet been uploaded
    return render_template('fraggle.html')



@app.route('/uploads/<filename>',methods=['GET', 'POST'])
def uploaded_file(filename):
    '''GET method runs the user training data to fit the RFC.
       POST will direct toward the hit display'''
    if request.method == 'POST':
        '''
        Accepts the file of high-throughput screen data and
        the user specified threshold and passes to the hits function.
        '''
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename_hts = secure_filename(file.filename)
            user_threshold = request.form['text']
            start_col = request.args.get('start_col')
            end_col = request.args.get('end_col')
            id_col = request.args.get('id_col')
            test = str(user_threshold)
            u_thres = float(test)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_hts))

            return redirect(url_for('hits', filename=filename_hts, user_thres = u_thres, start_col = start_col, end_col = end_col, id_col=id_col))

    elif request.method == 'GET':
        '''
        This function pulls in the uploaded train data. It seperates the features and y (hit score) from the training data and splits into a training and "test", really validation set. The minority class of the training data is oversampled to a ratio of .30 and a Random Forest Classifier is fit. The option exists to run a grid search over hyper parameters and will be implimented once I get on a larger computer. The threshold is explored and two possible suggestions are examined.
        '''
        hit_col = request.args.get('hit_col')
        start_col = request.args.get('start_col')
        end_col = request.args.get('end_col')
        # make a pd.dataframe of training data
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        features, yfill = proc.features_y_user(df, hit_col, start_col, end_col)
        # use all features and yfill (no NaNs, filled with 0)

        #train test split at 20%
        X_train, X_test, y_train, y_test = rf.train_test_split(features, yfill, test_size=0.20, random_state=1, stratify =yfill)

        #Optional: oversampling of minority class for training purposes
        X_train_over, y_train_over = proc.oversample(X_train,y_train, r = 0.3)
        rffit, y_predict = rf.randomforest(X_train_over, X_test, y_train_over, y_test)
        cm_base = confusion_matrix(y_test, y_predict)
        print('CM base threshold', cm_base)

        # Use below to run a grid search .... takes to long to work right now
        #rffit, y_predict, best_param = rf.randomforest(X_train.values, X_test, y_train.values, y_test, grid_search = 'small')

        #pickle the fit model for use with test data
        proc._pickle(rffit, 'RFC_fit.pkl')

        #set_threshold_recall is a function which determines the threshold to set such that recall is optimized (the median of the available thresholds that return the second best recall (not 1.0))

        precision_list, recall_list, median_recall_index, medianrecall_threshold = rf.set_threshold_recall(rffit, X_train_over, X_test, y_train_over, y_test)

        #print_threshold uses the trained model and the selected threshold (in this case recall optimized) to return listed statistics
        precision, recall, fpr, fpr_test, tpr_test, cm = rf.print_threshold(rffit, X_train_over, X_test, y_train_over, y_test, medianrecall_threshold)
        #pcm.plot_confusion_matrix_basic(cm, classes = ['Not a Hit', 'Hit'], name = 'recall_CM')
        pcm.seaborn_matrix(cm, 'recall_CM_seaborn')
        r_cm = pd.DataFrame(cm)
        proc._pickle(medianrecall_threshold, 'medianrecall_threshold.pkl')

        #make a pd.dataframe of the stats for display
        recall_opt_stats = pd.DataFrame([[format(medianrecall_threshold, '.2f'),format(recall, '.2f'), format(fpr, '.2f'), format(precision, '.2f'), ]], columns = ['Suggested Threshold','True Positive Rate (Recall)', 'False Positive Rate (Fall-out)','Precision'])

        # repeat the threshold selection process for precision optimization
        p_precision, p_recall, p_median_precision, threshold_precision = rf.set_threshold_precision(rffit, X_train_over, X_test, y_train_over, y_test)
        p_precision, p_recall, p_fpr, p_fpr_test, p_tpr_test, p_cm = rf.print_threshold(rffit, X_train_over, X_test, y_train_over, y_test, threshold_precision)
        #pcm.plot_confusion_matrix_basic(p_cm, classes = ['Not a Hit', 'Hit'], name = 'precision_CM')
        pcm.seaborn_matrix(p_cm, 'precision_CM_seaborn')
        p_cm = pd.DataFrame(p_cm)
        precision_opt_stats = pd.DataFrame([[format(threshold_precision, '.2f'),format(p_recall, '.2f'), format(p_fpr, '.2f'), format(p_precision, '.2f'), ]], columns = ['Suggested Threshold','True Positive Rate (Recall)', 'False Positive Rate (Fall-out)','Precision'])

        #produce a ROC plot
        test_prob = rffit.predict_proba(X_test)
        hts_a.plot_features(features, rffit, 'Train',n=10)
        roc.plot_roc(X_train.values, y_train.values, y_test, test_prob, 'new_Roc_train', RandomForestClassifier,  max_depth = 10, max_features= 30, min_samples_leaf= 2, min_samples_split = 2)
        feature_description = rf.plot_features(features, rffit, 'Identifier',n=10)
        #option for oversampled data
        #roc.plot_roc(X_train_over, y_train_over, y_test, test_prob, 'Test', RandomForestClassifier,  max_depth = 10, max_features= 30, min_samples_leaf= 2, min_samples_split = 2)
        #roc.simple_roc(y_test, test_prob, 'ROC_RFC')
        pd.set_option('display.max_colwidth', -1)

        #retrain the model with all data, & oversamplint of minority class
        #X_all_over, y_all_over = proc.oversample(features, y, r = 0.3)
        rffit.fit(features, yfill)
        proc._pickle(rffit, 'RFC_fit_all.pkl')
        # modified to run model trained on all fragment screen (not just training set)

        return render_template("rock_two.html", data_recall_opt=recall_opt_stats.to_html(index = False, classes = "data_recall_opt"),data_precision_opt=precision_opt_stats.to_html(index = False, classes = "data_precision_opt"), rocname = 'Test',f_descrip = feature_description.to_html(index = False, classes = "f_descrip"), recall_cm = r_cm.to_html(classes = "cm"), precision_cm = p_cm.to_html(classes = "cm"))


@app.route('/hits', methods=['GET', 'POST'])
def hits():
    '''
    If method is GET will put the uploaded high-throughput library featues into
    the trained model and display the results with the user specified threshold.
    If POST will accept a file with the hits scorred for the test data (high-throughput screen).
    A comparison of how FraggleRockHits did compared to this upload will be displayed.
    '''
    if request.method == 'POST':
        print('post method requested')
        print('submit equaled Validate')
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename_scorred = secure_filename(file.filename)
            #user_threshold = request.args.get['user_thres']
            hit_col = request.form['hit']
            id_col = request.form['id']
            #test = str(user_threshold)
            #u_thres = float(test)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_scorred))
            df_scorred = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename_scorred))
            hits_scorred, ids_scorred = proc.id_scores(df_scorred, id_col, hit_col)
            rffit = proc._unpickle('RFC_fit_all.pkl')
            features = proc._unpickle('features.pkl')
            u_thres = proc._unpickle('u_thres.pkl')
            print('u_thres')
            print(u_thres)
            y_proba = rffit.predict_proba(features)
            roc.plotHTS_roc(features, hits_scorred, y_proba, 'ROC_HTS')
            precision, recall, fpr, fpr_test, tpr_test, cm = hts_a.print_threshold(rffit, features, hits_scorred, float(u_thres))
            pcm.seaborn_matrix(cm, name = 'HTS_CM_seaborn')
            cm = pd.DataFrame(cm)
            recall_opt_stats = pd.DataFrame([[format(float(u_thres), '.2f'),format(recall, '.2f'), format(fpr, '.2f'), format(precision, '.2f'), ]], columns = ['Selected Threshold','True Positive Rate (Recall)', 'False Positive Rate (Fall-out)','Precision'])
            top50 = proc._unpickle('top50.pkl')
            dfthreshold = pd.DataFrame(['User Threshold', format(float(u_thres), '.2f')])

            return render_template('validate.html', articles = articles, data = recall_opt_stats.to_html(index = False, classes = "data"), hits = top50.to_html(index = False, classes = "table table-condensed"), thresh = dfthreshold.to_html(index= False, header = False, classes = "data"))
            #return redirect(url_for('hits', filename=filename_hts, user_thres = u_thres, start_col = start_col, end_col = end_col, id_col=id_col))

    elif request.method == 'GET':
        '''
        Takes in the uploaded high-throughput library and the user specified columns.
        Stores the scored library and displayes the top 50 predicted hits.
        '''
        start_col = request.args.get('start_col')
        end_col = request.args.get('end_col')
        id_col = request.args.get('id_col')
        u_thres = request.args.get('user_thres')
        filename_hts = request.args.get('filename')
        #This is used for validation (where y values are known)
        #features, y, ids = proc.features_yfill_HTS(df)
        #features, ids = proc.features_HTS_user(df)
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename_hts))
        features, ids = proc.features_HTS_user(df, start_col, end_col, id_col)
        rffit = proc._unpickle('RFC_fit.pkl')

        mrecall_threshold = float(u_thres)
        scored_hts = hts_a.score_HTS(rffit, features, ids, mrecall_threshold)
        top50 = scored_hts.iloc[:50,:]
        proc._pickle(top50, 'top50.pkl')
        dfthreshold = pd.DataFrame(['User Threshold', mrecall_threshold])
        scored_hts.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'scored_hts'))
        proc._pickle(features, 'features.pkl')
        proc._pickle(u_thres, 'u_thres.pkl')


        return render_template('hit.html', articles = articles, hits = top50.to_html(index = False, classes = "table table-condensed"), thresh = dfthreshold.to_html(index= False, header = False, classes = "data") )

@app.route('/about')
def articles():
    return render_template('about.html', articles = articles)

@app.route('/download',methods=['GET', 'POST'])
    '''Sending the scored high-throughput library data'''
def download_file():
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               'scored_hts', as_attachment=True)

if __name__ == '__main__':
#    app.run(host = '0.0.0.0', port =8105, threaded = True, debug=True)
     app.run(host = '0.0.0.0', port =80, threaded = True, debug=True)
