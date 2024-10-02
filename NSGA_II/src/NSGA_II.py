# -*- coding: utf-8 -*-
# @Author  : Han Li
# @Time    : 2024/9/11 18.11
# @File    : NSGA_II.py


import logging
import sys

from model.NSGA_II_model import NAGA_II
import argparse
import numpy as np

logging.basicConfig(
        filename='NAGA_II.log',
        format='[%(asctime)s][%(filename)s][%(levelname)s][%(message)s]',
        level=logging.INFO,
        filemode="w")


def get_args():
    parse = argparse.ArgumentParser(description="random forest classification",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("-ipd", "--initial_pop_data",
                       dest="initial_pop_data", type=str, default="../data/input.csv",
                       help="Initial population")
    parse.add_argument("-mag", "--max_gen",
                       dest="max_gen", type=int, default=100,
                       help="maximum number of iterations")
    parse.add_argument("-opd", "--out_pop_data", dest="out_pop_data",
                       type=str, default="../data/naga_output.csv", help="optimized result")

    return parse.parse_args()


args = get_args()

input_data=args.initial_pop_data
max_gen=args.max_gen
output_path=args.out_pop_data

try:
    inital_data = np.loadtxt(input_data, dtype=float, delimiter=',')
    inital_list=[]
    for i in range(len(inital_data)):
        inital_list.append(float(inital_data[i]))
except Exception as input_err:
    logging.error("file not found or failed to load file！")
    logging.error(input_err)
    sys.exit(1)
else:
    logging.info("Initial data load successfully.")
    try:
        max_gen=int(max_gen)
        result_path = str(output_path)
    except Exception as parameter_error:
        logging.error("Parameter loading error！")
        logging.error(parameter_error)
        sys.exit(1)
    else:
        logging.info("Parameter load successfully.")
        try:
            naga=NAGA_II(inital_data=inital_list,max_gen=max_gen)
        except Exception as Initial:
            logging.error("NAGA initialization failed！")
            logging.error(Initial)
            sys.exit(1)
        else:
            logging.info("NAGA initialization successfully.")
            try:
                naga.run()
            except Exception as run_error:
                logging.error("NAGA calculation error！")
                logging.error(run_error)
                sys.exit(1)
            else:
                logging.info("NAGA calculation successfully.")
                try:
                    naga.result(output_path)
                except Exception as out_error:
                    logging.error("Result write error!")
                    logging.error(out_error)
                    sys.exit(1)
                else:
                    logging.info("Result write successfully.")


