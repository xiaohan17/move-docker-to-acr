# -*- coding: utf-8 -*-
# @Author  : Tianming Chao
# @Time    : 2022/7/17 18:51
# @File    : svm.py

from model.SupportVectorMachines import SupportVectorMachines
import argparse
import pandas as pd
import numpy as np
import sys
import os
import logging

logging.basicConfig(filename='svm.log',
                    format='[%(asctime)s][%(filename)s][%(levelname)s][%(message)s] ',
                    level=logging.INFO,
                    filemode="w")


def get_args():
    parse = argparse.ArgumentParser(description="C-Support Vector Classification.",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("-train_data", "--train_data",
                       type=str, default="../data/train_data.csv")
    parse.add_argument("-train_label", "--train_label",
                       type=str, default="../data/train_label.csv")
    parse.add_argument("-test_data", "--test_data",
                       type=str, default="../data/test_data.csv")
    parse.add_argument("-test_label", "--test_label",
                       type=str, default="../data/test_label.csv")
    parse.add_argument("-C", "--C",
                       type=float, default=1.0, help="Regularization parameter")
    parse.add_argument("-k", "--kernel",
                       type=str, default="rbf", help="Specifies the kernel type to be used in the algorithm")
    parse.add_argument("-d", "--degree",
                       type=int, default=3, help="Degree of the polynomial kernel function (‘poly’)")
    parse.add_argument("-g", "--gamma",
                       type=str, default="scale", help="Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’")
    parse.add_argument("-c", "--coef0",
                       type=float, default=0.0,
                       help="Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’")
    parse.add_argument("-t", "--tol", type=float, default=1e-3)
    parse.add_argument("-cache_size", "--cache_size", type=float, default=200.0)
    parse.add_argument("-m", "--max_iter", type=int, default=-1)
    parse.add_argument("-decision_function_shape", "--decision_function_shape", type=str, default='ovr')
    parse.add_argument("-output", "--output",
                       type=str, default="./c_support_vector_classification.csv", help="output result path")
    return parse.parse_args()


def main():
    args = get_args()

    try:
        train_data = np.loadtxt(args.train_data, delimiter=',')
        train_label = np.loadtxt(args.train_label, delimiter=',')
        test_data = np.loadtxt(args.test_data, delimiter=',')
        test_label = np.loadtxt(args.test_label, delimiter=',')
    except Exception:
        logging.error("file not found or failed to load file.")
        sys.exit(1)
    else:
        logging.info("input data load successfully.")
        try:
            args.C = float(args.C)
            args.degree = int(args.degree)
            args.coef0 = float(args.coef0)
            args.tol = float(args.tol)
            args.cache_size = float(args.cache_size)
            args.max_iter = int(args.max_iter)
        except Exception:
            logging.error("wrong or missing parameter value.")
            sys.exit(1)
        else:
            logging.info("parameter load successfully.")
            try:
                predictor = SupportVectorMachines(train_data=train_data,
                                                  train_label=train_label,
                                                  C=args.C,
                                                  kernel=args.kernel,
                                                  degree=args.degree,
                                                  gamma=args.gamma,
                                                  coef0=args.coef0,
                                                  tol=args.tol,
                                                  cache_size=args.cache_size,
                                                  max_iter=args.max_iter,
                                                  decision_function_shape=args.decision_function_shape)
                predictor.run()
            except Exception:
                logging.error("model calculation failed.")
                sys.exit(1)
            else:
                try:
                    svc_score = predictor.svc_score(test_data=test_data, test_label=test_label)
                    logging.info('The mean accuracy on the given test data and labels is %s', svc_score)
                except Exception:
                    logging.error("model calculation failed.")
                    sys.exit(1)
                else:
                    logging.info("model calculation succeeded.")
                    try:
                        predict_label = predictor.predict(test_data)
                        predictor.out(res=predict_label, out_path=args.output)
                    except Exception:
                        logging.error("output data export failed.")
                        sys.exit(1)
                    else:
                        logging.info("output data export successfully.")


if __name__ == '__main__':
    main()
