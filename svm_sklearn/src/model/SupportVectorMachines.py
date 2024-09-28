# -*- coding: utf-8 -*-
# @Author  : Tianming Zhao
# @Time    : 2022/7/17 18:51
# @File    : SupportVectorMachines.py

import logging
from sklearn.svm import SVC
import numpy as np
import pandas as pd


class SupportVectorMachines:
    def __init__(self,
                 train_data: np.ndarray,
                 train_label: np.ndarray,
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 degree: int = 3,
                 gamma: str = 'scale',
                 coef0: float = 0.0,
                 shrinking: bool = True,
                 probability: bool = False,
                 tol: float = 1e-3,
                 cache_size: float = 200,
                 verbose: bool = False,
                 max_iter: int = -1,
                 decision_function_shape: str = 'ovr',
                 break_ties: bool = False):
        self._train_data = train_data
        self._train_label = train_label

        self._C = C
        self._kernel = kernel
        self._degree = degree
        self._gamma = gamma
        self._coef0 = coef0
        self._shrinking = shrinking
        self._probabilitybool = probability
        self._tol = tol
        self._cache_size = cache_size
        self._verbose = verbose
        self._max_iterint = max_iter
        self._decision_function_shape = decision_function_shape
        self._break_ties = break_ties

        self._svm = c_support_vector_classification(C=self._C,
                                                    kernel=self._kernel,
                                                    degree=self._degree,
                                                    gamma=self._gamma,
                                                    coef0=self._coef0,
                                                    shrinking=self._shrinking,
                                                    probability=self._probabilitybool,
                                                    tol=self._tol,
                                                    cache_size=self._cache_size,
                                                    verbose=self._verbose,
                                                    max_iter=self._max_iterint,
                                                    decision_function_shape=self._decision_function_shape,
                                                    break_ties=self._break_ties)

    def run(self):
        predictor = self._svm.fit(self._train_data, self._train_label)
        return predictor

    def out(self, res, out_path):
        if type(res) == np.ndarray:
            np.savetxt(fname=out_path, X=res, fmt='%.5f', delimiter=',')
        elif type(res) == pd.DataFrame:
            res.to_csv(path_or_buf=out_path)
        return True

    def svc_score(self, test_data: np.ndarray, test_label: np.ndarray):
        """
        Return the mean accuracy on the given test data and labels.
        @param X: array-like of shape (n_samples, n_features)
        @param y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        @param sample_weight: array-like of shape (n_samples,), default=None
        """
        test_score = self._svm.score(test_data, test_label, sample_weight=None)
        test_score2 = str(test_score)
        return test_score

    def predict(self, data: np.ndarray = None):
        """
        Perform regression on samples in X.
        @param X : {array-like, sparse matrix} of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).
        @param data: data to make predictions
        """
        logging.info('Perform classification on samples in X.For an one-class model, +1 or -1 is returned.')
        predict_label = self._svm.predict(data)
        return predict_label


def main():
    pass


if __name__ == '__main__':
    main()


def c_support_vector_classification(C=1.0, kernel='rbf', degree=3, gamma='scale',
                                    coef0=0.0, shrinking=True, probability=False,
                                    tol=1e-3, cache_size=200.0,
                                    verbose=False, max_iter=-1, decision_function_shape='ovr',
                                    break_ties=False):
    return SVC(C=C,
               kernel=kernel,
               degree=degree,
               gamma=gamma,
               coef0=coef0,
               shrinking=shrinking,
               probability=probability,
               tol=tol,
               cache_size=cache_size,
               verbose=verbose,
               max_iter=max_iter,
               decision_function_shape=decision_function_shape,
               break_ties=break_ties)
