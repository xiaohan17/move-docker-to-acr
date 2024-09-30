# -*- coding:utf-8 -*-
"""
# @Author  :Shan Huang
# @Time    :2023/03/16 22:26
# @File    :multiple_layer_perceptron.py
"""
import src.LoadData
import argparse
import logging
import os
import sys
from src.TrainDataSet import IntegratedMLP
from src.LoadData import LoadData
from src.multiple_layer_perceptron_class import MLP
import argparse
import logging
import sys
import os

logging.basicConfig(filename='multiple_layer_perceptron.log',
                    format='[%(asctime)s][%(filename)s][%(levelname)s][%(message)s] ',
                    level=logging.INFO,
                    filemode="w")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', '-dr', type=str, default='data')
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--num_workers', '-n', type=int, default=4)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1)
    parser.add_argument('--num_epochs', '-ne', type=int, default=3)
    parser.add_argument('--num_hidden_layers', '-nhl', type=int, default=1)
    parser.add_argument('--num_hidden_units', '-nhu', type=int, default=256)
    parser.add_argument('--accuracy_file', '-a', type=str, default='accuracy.txt')
    parser.add_argument('--mean', '-mean', type=str, default='0.0')
    parser.add_argument('--std', '-std', type=str, default='0.1')
    parser.add_argument('--model_dir', '-m', type=str, default='./src')
    parser.add_argument('--predict_dir', '-pd', type=str, default='data/predict/')
    parser.add_argument('--loaded_model', '-ld', type=str, default='data/epoch19.pth')
    parser.add_argument('--result_dir', '-rd', type=str, default='./result')
    cfg = parser.parse_args()

    try:
        if not os.path.exists(cfg.data_root):
            os.makedirs(cfg.data_root)
        if not os.path.exists(cfg.result_dir):
            os.makedirs(cfg.result_dir)
        if not os.path.exists(cfg.result_dir):
            os.makedirs(cfg.result_dir)
        cfg.batch_size = int(cfg.batch_size)
        data = LoadData(cfg.batch_size, resize=None, root=cfg.data_root, num_workers=cfg.num_workers)
        data.load_data_mnist()
    except Exception:
        logging.error("file not found or failed to load file.")
        sys.exit(1)
    else:
        logging.info("input data load successfully.")
        try:
            cfg.learning_rate = float(cfg.learning_rate)
            cfg.num_epochs = int(cfg.num_epochs)
            cfg.num_workers = int(cfg.num_workers)
            cfg.num_hidden_layers = int(cfg.num_hidden_layers)
            cfg.num_hidden_units = int(cfg.num_hidden_units)
            cfg.mean = float(cfg.mean)
            cfg.std = float(cfg.std)
        except Exception:
            logging.error("wrong or missing parameter value.")
            sys.exit(1)
        else:
            logging.info("parameter load successfully.")
            net = IntegratedMLP(cfg=cfg, train_data=data.train_iter, test_data=data.test_iter)
            net.train()
            try:
                pass
                # net = IntegratedMLP(cfg=cfg, train_data=data.train_iter, test_data=data.test_iter)
                # net.train()
            except Exception:
                logging.error("model calculation failed.")
                sys.exit(1)
            else:
                try:
                    net.save_evaluate()
                    logging.info("model calculation succeeded.")

                except Exception:
                    logging.error("model calculation failed.")
                    sys.exit(1)
                else:
                    try:
                        net.predict()
                    except Exception:
                        logging.error("output data export failed.")
                        sys.exit(1)
                    else:
                        logging.error("output data export successfully.")


if __name__ == '__main__':
    main()
