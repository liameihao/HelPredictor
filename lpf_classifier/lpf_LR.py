from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def scale_data(X_train, X_test):
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = min_max_scaler.fit(X_train)
    X_train_ = scaler.transform(X_train)
    X_test_ = scaler.transform(X_test)
    return X_train_, X_test_


def LR(X_train, y_train, X_test, y_test, njob,
       get_model=False, number=None, method=None):
    # classifier
    reg = LogisticRegression(solver='lbfgs', multi_class="ovr", max_iter=3000)
    # parameters selection
    parameters = {'C': [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]}
    # grid search
    clf = GridSearchCV(reg, parameters, cv=5, scoring="accuracy",
                       return_train_score=False, n_jobs=njob)
    # train
    clf.fit(X_train, y_train)
    # get best accuracy of train data
    train_accuracy = clf.best_score_
    # predict test data
    predict_test = clf.best_estimator_.predict(X_test)
    # get accuracy of test data
    test_accuracy = accuracy_score(y_test, predict_test)
    if get_model:
        model_name = "{0}_{1}_LR.joblib".format(number, method)
        dump(clf, model_name)
    return train_accuracy, test_accuracy, clf.best_params_


def LR_overall(X_train, y_train, X_test, y_test,
               step_start, step_end, step, method, njobs):
    accuracy_data = pd.DataFrame(columns=["feature number", "train_accuracy",
                                          "test_accuracy", "parameters"])

    for i in range(step_start, step_end, step):
        X_train_ = X_train[:, :i]
        X_test_ = X_test[:, :i]
        X_train_, X_test_ = scale_data(X_train_, X_test_)
        feature_number = X_train_.shape[1]
        train_accuracy, test_accuracy, best_params_ = LR(X_train_, y_train, X_test_, y_test, njobs)
        accuracy_data.loc[i] = feature_number, train_accuracy, test_accuracy, best_params_
    accuracy_data.to_csv("{0}-{1}_{2}_LR_accuracy.csv".format(step_start, step_end - 1, method), index=False)
    print("the result: {0}-{1}_{2}_LR_accuracy.csv".format(step_start, step_end - 1, method))
