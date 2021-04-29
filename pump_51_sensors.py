import os, gc, matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed

from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score, precision_score, recall_score, plot_confusion_matrix

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers

##################################################################################################################################################################################################################################################################################################################################################################


def plot_learning_curve(estimator, title, X, y, score_met='average_precision', ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 7)):

    fig, axes = plt.subplots(1, 1, dpi=dpi_setting)

    axes.set_title("Learning Curve for "+title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel(score_met)

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, scoring=score_met, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=False)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes.legend(loc="best")

    return plt


def data_preprocessing(data_loaded, ml_method='Supervised', fraction_sample=0.5, multiplier_factor_sample=1, dx_val=1.0, do_derivative=False, do_std_scl=True, do_pca=False, unsupervised_train_onfraud=False, do_over_sampling=False):

    data_csv = data_loaded

    anomaly_data = data_csv[data_csv['Class'] == 1]
    normal = data_csv[data_csv['Class'] == 0]
    anomaly_data_sampled = anomaly_data.sample(frac=fraction_sample)
    normal_sampled = normal.sample(int(multiplier_factor_sample*anomaly_data_sampled.shape[0]))
    data_csv = data_csv.drop(anomaly_data_sampled.index)
    data_csv = data_csv.drop(normal_sampled.index)
    data_csv = data_csv.reset_index(drop=True)

    anomaly_data_sampled = anomaly_data_sampled.interpolate(method='linear', axis=1)
    anomaly_data_sampled = anomaly_data_sampled.fillna(anomaly_data_sampled.mean())
    normal_sampled = normal_sampled.interpolate(method='linear', axis=1)
    normal_sampled = normal_sampled.fillna(normal_sampled.mean())
    data_csv = data_csv.interpolate(method='linear', axis=1)
    data_csv = data_csv.fillna(data_csv.mean())

    if do_over_sampling in [True]:
        smote_over_sample = BorderlineSMOTE(sampling_strategy='minority', random_state=42, k_neighbors=5, n_jobs=-1, m_neighbors=10, kind='borderline-1')
        train_concat = pd.concat([normal_sampled,anomaly_data_sampled],axis=0)
        train_classes = train_concat['Class']
        train_concat = train_concat.drop(['Class'],axis=1)
        X_resampled, y_resampled = smote_over_sample.fit_resample(train_concat, train_classes)
        train_concat = pd.concat([X_resampled, y_resampled],axis=1)
        anomaly_data_sampled = train_concat[train_concat['Class'] == 1]
        normal_sampled = train_concat[train_concat['Class'] == 0]

    if ml_method in ['Supervised']:

        X_train = pd.concat([normal_sampled,anomaly_data_sampled],axis=0)
        y_train = X_train['Class']
        X_train = X_train.drop(['Class'],axis=1)
        y_valid = data_csv['Class']
        data_csv = data_csv.drop(['Class'],axis=1)
        X_valid = data_csv

    elif ml_method in ['Unsupervised']:

        if unsupervised_train_onfraud in [False]: #Training unsupervised model on normal transactions samples
            X_train = normal_sampled
            y_train = X_train['Class']
            X_train = X_train.drop(['Class'],axis=1)
            X_valid = pd.concat([data_csv, anomaly_data_sampled], axis=0)
            y_valid = X_valid['Class']
            X_valid = X_valid.drop(['Class'],axis=1)

        elif unsupervised_train_onfraud in [True]: #Training unsupervised model on fraudulent transactions samples
            X_train = anomaly_data_sampled
            y_train = X_train['Class']
            X_train = X_train.drop(['Class'],axis=1)
            X_valid = pd.concat([data_csv, normal_sampled], axis=0)
            y_valid = X_valid['Class']
            X_valid = X_valid.drop(['Class'],axis=1)

    elif ml_method in ['Semisupervised']:

        if unsupervised_train_onfraud in [False]: #Training unsupervised model on normal transactions samples
            X_train = normal #normal_sampled
            y_train = X_train['Class']
            X_train = X_train.drop(['Class'],axis=1)
            X_valid = anomaly_data_sampled #pd.concat([data_csv, anomaly_data_sampled], axis=0)
            y_valid = X_valid['Class']
            X_valid = X_valid.drop(['Class'],axis=1)

        elif unsupervised_train_onfraud in [True]: #Training unsupervised model on fraudulent transactions samples
            X_train = anomaly_data_sampled
            y_train = X_train['Class']
            X_train = X_train.drop(['Class'],axis=1)
            X_valid = normal #normal_sampled #pd.concat([data_csv, normal_sampled], axis=0)
            y_valid = X_valid['Class']
            X_valid = X_valid.drop(['Class'],axis=1)

    if do_std_scl in [True]:
        stdscl = StandardScaler()
        X_train = stdscl.fit_transform(X_train)
        X_valid = stdscl.transform(X_valid)

    if do_derivative in [True]:
        X_train = np.gradient(X_train, dx_val, axis=1)
        X_valid = np.gradient(X_valid, dx_val, axis=1)

    if do_pca in [True]:
        pca_obj = PCA(n_components=0.95, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=42)
        print("Train shape before PCA", X_train.shape)
        X_train = pca_obj.fit_transform(X_train)
        X_valid = pca_obj.transform(X_valid)
        print("Train shape after PCA", X_train.shape)

    return X_train, X_valid, y_train, y_valid


def autoencoder_supervised_classifier(X_normal, X_anomaly_data):

    ## input layer
    input_layer = Input(shape=(X_normal.shape[1],))

    ## encoding part
    encoded = Dense(200, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoded = Dense(50, activation='relu')(encoded)
    #encoded = Dense(25, activation='relu')(encoded)

    ## decoding part
    #decoded = Dense(25, activation='tanh')(encoded)
    decoded = Dense(50, activation='tanh')(encoded)
    decoded = Dense(200, activation='tanh')(decoded)

    ## output layer
    output_layer = Dense(X_normal.shape[1], activation='relu')(decoded)

    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer="adadelta", loss="mse")
    autoencoder.fit(X_normal, X_normal, batch_size = 256, epochs = 10, shuffle = True, validation_split = 0.20, verbose=0)

    hidden_representation = Sequential()
    hidden_representation.add(autoencoder.layers[0])
    hidden_representation.add(autoencoder.layers[1])
    hidden_representation.add(autoencoder.layers[2])
    norm_hid_rep = hidden_representation.predict(X_normal)
    anomaly_data_hid_rep = hidden_representation.predict(X_anomaly_data)

    X_represent_transformed = np.append(norm_hid_rep, anomaly_data_hid_rep, axis = 0)
    y_normal = np.zeros(norm_hid_rep.shape[0])
    y_anomaly_datas = np.ones(anomaly_data_hid_rep.shape[0])
    y_represent_transformed = np.append(y_normal, y_anomaly_datas)

    return X_represent_transformed, y_represent_transformed


def scoring_metrics_calculation(ytrn_true, yvld_true, ytrn_pred, yvld_pred, ytrn_pred_proba, yvld_pred_proba):

    accur = accuracy_score(ytrn_true,ytrn_pred)
    precs = precision_score(ytrn_true, ytrn_pred, average='weighted')
    recal = recall_score(ytrn_true, ytrn_pred, average='weighted')
    auprc = average_precision_score(ytrn_true,ytrn_pred_proba)
    conmat = confusion_matrix(ytrn_true, ytrn_pred, normalize='all')
    tn, fp, fn, tp = conmat.ravel()

    score_dict_train = dict(accur=accur, precs=precs, recal=recal, auprc=auprc, tn=tn, fp=fp, fn=fn, tp=tp)

    accur = accuracy_score(yvld_true,yvld_pred)
    precs = precision_score(yvld_true, yvld_pred, average='weighted')
    recal = recall_score(yvld_true, yvld_pred, average='weighted')
    auprc = average_precision_score(yvld_true,yvld_pred_proba)
    conmat = confusion_matrix(yvld_true, yvld_pred, normalize='true')
    tn, fp, fn, tp = conmat.ravel()

    score_dict_valid = dict(accur=accur, precs=precs, recal=recal, auprc=auprc, tn=tn, fp=fp, fn=fn, tp=tp)

    return score_dict_train, score_dict_valid


def machine_learning_model(estimator, xtrn, xvld, ytrn, yvld, ml_method='Supervised', n_splits=3, steps=1, score_met='average_precision', param_grid=None, do_gscv=False, do_recursive_elimination=False, plot_learn_curve=False, plot_con_matrix=False):

    cvld = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, train_size=None, random_state=42)

    if do_recursive_elimination in [True]:
        # estimator.fit(xtrn, ytrn)
        print("\nRecursive Feature Eliminiation.....")
        rfecv = RFECV(estimator, step=steps, min_features_to_select=1, cv=cvld, scoring=score_met, verbose=0, n_jobs=-1)
        try:
            X_train_new = rfecv.fit_transform(xtrn, ytrn)
        except RuntimeError:
            logreg_rf = LogisticRegression(C=1.0, penalty='l2', dual=False, tol=0.0001, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=42, solver='lbfgs', max_iter=1000, multi_class='auto', verbose=0, warm_start=False, n_jobs=-1, l1_ratio=None)
            rfecv = RFECV(logreg_rf, step=steps, min_features_to_select=1, cv=cvld, scoring=score_met, verbose=0, n_jobs=-1)
            X_train_new = rfecv.fit_transform(xtrn, ytrn)
        print("Optimal features: %d" % rfecv.n_features_)
        X_valid_new = rfecv.transform(xvld)
        xtrn = X_train_new
        xvld = X_valid_new

    if do_gscv in [True]:
        print("\nGrid Search CV.....")
        print(param_grid)
        gscv = GridSearchCV(estimator, param_grid=param_grid, scoring=score_met, n_jobs=-1, refit=True, cv=cvld, verbose=0, pre_dispatch='2*n_jobs', return_train_score=False)
        gscv.fit(xtrn, ytrn)
        ytrn_pred = gscv.predict(xtrn)
        yvld_pred = gscv.predict(xvld)
        ytrn_pred_proba_both = gscv.predict_proba(xtrn)
        ytrn_pred_proba = ytrn_pred_proba_both[:,1]
        yvld_pred_proba_both = gscv.predict_proba(xvld)
        yvld_pred_proba = yvld_pred_proba_both[:,1]
        print(gscv.best_estimator_)
        print(gscv.best_score_)
        print(gscv.best_params_)
        estimator = gscv.best_estimator_

    if ml_method in ['Supervised']:
        estimator.fit(xtrn, ytrn)
        ytrn_pred = estimator.predict(xtrn)
        yvld_pred = estimator.predict(xvld)
        ytrn_pred_proba_both = estimator.predict_proba(xtrn)
        ytrn_pred_proba = ytrn_pred_proba_both[:,1]
        yvld_pred_proba_both = estimator.predict_proba(xvld)
        yvld_pred_proba = yvld_pred_proba_both[:,1]

    elif ml_method in ['Unsupervised']:
        estimator.fit(xtrn)
        ytrn_pred = estimator.predict(xtrn)
        yvld_pred = estimator.predict(xvld)
        ytrn_pred = [1 if l == -1 else 0 for l in ytrn_pred]
        yvld_pred = [1 if l == -1 else 0 for l in yvld_pred]
        ytrn_pred_proba = ytrn_pred
        yvld_pred_proba = yvld_pred

    elif ml_method in ['Semisupervised']:
        X_represent_transformed, y_represent_transformed = autoencoder_supervised_classifier(xtrn, xvld)
        X_train, X_valid, y_train, y_valid = train_test_split(X_represent_transformed, y_represent_transformed, test_size=0.25)
        estimator.fit(X_train, y_train)
        ytrn_pred = estimator.predict(X_train)
        yvld_pred = estimator.predict(X_valid)
        ytrn_pred_proba_both = estimator.predict_proba(X_train)
        ytrn_pred_proba = ytrn_pred_proba_both[:,1]
        yvld_pred_proba_both = estimator.predict_proba(X_valid)
        yvld_pred_proba = yvld_pred_proba_both[:,1]
        ytrn = y_train
        yvld = y_valid

    score_dict_train, score_dict_valid = scoring_metrics_calculation(ytrn, yvld, ytrn_pred, yvld_pred, ytrn_pred_proba, yvld_pred_proba)

    if plot_learn_curve in [True]:
        print("\nPlotting Learning Curve.....")
        ### Learning Curves
        title = str(estimator)
        plot_learning_curve(estimator, title, xtrn, ytrn, ylim=(0.0, 1.1), cv=cvld, n_jobs=-1)
        plt.show()

    if plot_con_matrix in [True]:
        print("Plotting Confusion Matrix.....")
        ### Confusion Matrix
        plot_confusion_matrix(estimator, xvld, yvld, labels=None, sample_weight=None, normalize='true', display_labels=None, include_values=True, xticks_rotation='horizontal', values_format=None, cmap='viridis', ax=None)
        plt.show()

    return score_dict_train, score_dict_valid


##################################################################################################################################################################################################################################################################################################################################################################

startTime= datetime.now()
data_path = '/media/ashutosh/Computer Vision/Predictive_Maintenance/Pump_Sensor_Kaggle'
# data_path = 'E:\Predictive_Maintenance\Bank_Loan_data_Kaggle'
data_csv = pd.read_csv(data_path+"//sensor.csv")
num_samples = data_csv['Unnamed: 0']
data_csv = data_csv.drop(['Unnamed: 0'], axis=1)
data_csv = data_csv.drop(['timestamp'], axis=1)
data_csv = data_csv.drop(['sensor_15'], axis=1)
data_csv = data_csv.drop(['sensor_50'], axis=1)



# data_csv['damage_duration'] = 0.0
# for idx in range(data_csv.shape[0]):

#     if data_csv.loc[idx, 'machine_status'] in ['BROKEN', 'RECOVERING']:
#         data_csv.loc[idx, 'damage_duration'] = data_csv.loc[idx-1, 'damage_duration'] + 1.0

data_csv['Class'] = np.array([0 if x == 'NORMAL' else 1 for x in data_csv['machine_status']], dtype=int)
data_csv_sensors = data_csv.drop(['machine_status'], axis=1)

#data_csv_sensors = data_csv_sensors.interpolate(method='linear', axis=1)
#data_csv_sensors = data_csv_sensors.fillna(method='ffill')

dpi_setting = 120
plt.rcParams.update({'font.size': 7})
bins_use = 500

Log_Reg = LogisticRegression(C=1.0, penalty='l2', dual=False, tol=0.0001, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=42, solver='lbfgs', max_iter=1000, multi_class='auto', verbose=0, warm_start=False, n_jobs=-1, l1_ratio=None)
SVC_Linear = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo', break_ties=False, random_state=42)
SVC_RBF = SVC(C=100.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo', break_ties=False, random_state=42)
Decision_Tree = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=16, min_samples_split=26, min_samples_leaf=2, max_features=None, random_state=42, class_weight=None)
Random_Forest = RandomForestClassifier(n_estimators=90, criterion='entropy', max_depth=10, min_samples_split=8, min_samples_leaf=2, max_features='sqrt', bootstrap=True, oob_score=False, class_weight=None, n_jobs=-1, random_state=42)
XGBoost = XGBClassifier(n_estimators=100, max_depth=5, min_child_weight=2, max_delta_step=8, learning_rate=0.1, gamma=0.1, objective='binary:logistic', scale_pos_weight=1, base_score=0.85, missing=None, n_jobs=-1, nthread=-1, random_state=42, seed=42, silent=True, subsample=1, verbosity=0)
LightGBM = LGBMClassifier(n_estimators=115, num_leaves=65, max_depth=15, min_child_samples=40, learning_rate=0.1, boosting_type='gbdt', objective='binary', random_state=42, n_jobs=- 1, silent=True)
Naive_Bayes = GaussianNB(var_smoothing=1e0)
One_Class_SVM = OneClassSVM(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.05, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
Isolation_Forest = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=True, n_jobs=-1, random_state=42)
Auto_Enc_LogReg = Log_Reg
Auto_Enc_LightGBM = LightGBM

score_met = 'average_precision'
ml_method = 'Supervised'
fraction_sample = 0.5
multiplier_factor_sample = 1
n_splits = 5
steps = 1
dx_val=1.0
do_derivative = True
do_std_scl = True
do_pca = True
unsupervised_train_onfraud = False
do_gscv = False
do_over_sampling = False
do_recursive_elimination = False
plot_learn_curve = True
plot_con_matrix = True
param_grid = None

one_clf = SVC_RBF

print("\n.......................................................     Start     .......................................................\n")

gc.collect()

X_train, X_valid, y_train, y_valid = data_preprocessing(data_csv_sensors, ml_method=ml_method, unsupervised_train_onfraud=unsupervised_train_onfraud, fraction_sample=fraction_sample, multiplier_factor_sample=multiplier_factor_sample, do_over_sampling=do_over_sampling, dx_val=dx_val, do_derivative=do_derivative, do_std_scl=do_std_scl, do_pca=do_pca)

score_dict_train, score_dict_valid = machine_learning_model(one_clf, X_train, X_valid, y_train, y_valid, ml_method=ml_method, n_splits=n_splits, steps=steps, score_met=score_met, param_grid=param_grid, do_gscv=do_gscv, do_recursive_elimination=do_recursive_elimination, plot_learn_curve=plot_learn_curve, plot_con_matrix=plot_con_matrix)

print(X_train.shape)
print(X_valid.shape)

print("Training score:")
print(score_dict_train)
print("\nValidation score:")
print(score_dict_valid)

print("\n.......................................................      Done     .......................................................\n")
timeElapsed = datetime.now() - startTime
print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))

