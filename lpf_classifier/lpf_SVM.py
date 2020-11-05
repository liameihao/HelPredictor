from sklearn.svm import SVC
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


def SVM(X_train, y_train, X_test, y_test, njob,
        get_model=False, number=None, method=None):
    # classifier
    lpf_svm = SVC(decision_function_shape="ovo", random_state=0)
    # parameters selection
    C = []
    for i in range(-5, 15 + 1, 2):
        C.append(2**i)
    gamma = []
    for i in range(-15, 3 + 1, 2):
        gamma.append(2**i)
    parameters = {'C': C,
                  'gamma': gamma}

    # grid search
    clf = GridSearchCV(lpf_svm, parameters, cv=5, scoring="accuracy",
                       return_train_score=False, n_jobs=njob)
    # train
    clf.fit(X_train, y_train)
    # get best accuracy of train data
    train_accuracy = clf.best_score_
    # predict test data
    predict_test = clf.predict(X_test)
    # get accuracy of test data
    test_accuracy = accuracy_score(y_test, predict_test)
    if get_model:
        model_name = "{0}_{1}_SVM.joblib".format(number, method)
        dump(clf, model_name)
    return train_accuracy, test_accuracy, clf.best_params_


def SVM_overall(X_train, y_train, X_test, y_test,
               step_start, step_end, step, method, njobs):
    accuracy_data = pd.DataFrame(columns=["feature number", "train_accuracy",
                                          "test_accuracy", "parameters"])

    for i in range(step_start, step_end, step):
        X_train_ = X_train[:, :i]
        X_test_ = X_test[:, :i]
        X_train_, X_test_ = scale_data(X_train_, X_test_)
        feature_number = X_train_.shape[1]
        train_accuracy, test_accuracy, best_params_ = SVM(X_train_, y_train, X_test_, y_test, njobs)
        accuracy_data.loc[i] = feature_number, train_accuracy, test_accuracy, best_params_
    accuracy_data.to_csv("{0}-{1}_{2}_SVM_accuracy.csv".format(step_start, step_end - 1, method), index=False)
    print("the result: {0}-{1}_{2}_SVM_accuracy.csv".format(step_start, step_end - 1, method))
