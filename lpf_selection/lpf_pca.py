from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def scale_data(X_train):
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = min_max_scaler.fit(X_train)
    X_train_ = scaler.transform(X_train)
    return X_train_


def pca_selection(data, filename):
    X = data.iloc[:, 1:]
    X_s = scale_data(X)
    pca = PCA(n_components=1)
    pca.fit(X_s)
    pc1_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    pc1_FeatureScore = pd.DataFrame({'Feature': X.columns,
                                     'PC1_loading': pc1_loadings.T[0],
                                     'PC1_loading_abs': abs(pc1_loadings.T[0])})
    pc1_FeatureScore = pc1_FeatureScore.sort_values('PC1_loading_abs', ascending=False)
    pc1_FeatureScore.to_csv("{}_PCA_selection.csv".format(filename), index=False)
    data_train = data.reindex(['Label']+list(pc1_FeatureScore['Feature']), axis=1)
    data_train.to_csv("{}_PCA_data.csv".format(filename), index=None)
    return data_train
