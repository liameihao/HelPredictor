import pandas as pd
import numpy as np
import statsmodels.api as sm


def cal_cv2(data, filename):
    b = data.iloc[:, 1:].T.values
    c = b.copy()
    #
    means = np.mean(c, axis=1)
    variance = np.var(c, axis=1)
    cv2 = variance/means**2
    #
    minMeanForFit = np.quantile(means[np.where(cv2 > 0.5)], 0.95)
    useForFit = means >= minMeanForFit
    gamma_model = sm.GLM(cv2[useForFit], np.array([np.repeat(1, means[useForFit].shape[0]), 1/means[useForFit]]).T,
                         family=sm.families.Gamma(link=sm.genmod.families.links.identity))
    gamma_results = gamma_model.fit()
    a0 = gamma_results.params[0]
    a1 = gamma_results.params[1]
    afit = a1/means + a0
    varFitRatio = variance / (afit*(means**2))
    cv2_score = pd.DataFrame({'Feature': data.columns[1:], "cv2_score": varFitRatio})
    cv2_score = cv2_score.sort_values('cv2_score', ascending=False)
    cv2_score.to_csv("{}_cv2.csv".format(filename), index=False)
    data_train = data.reindex(['Label']+list(cv2_score['Feature']), axis=1)
    data_train.to_csv("{}_cv2_data.csv".format(filename), index=None)
    return data_train
