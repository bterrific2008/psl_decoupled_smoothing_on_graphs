from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot as plt
import random as rand
import numpy as np
import os

debug = False


def read_predictions(method, data_nm, random_seed, pct_lbl, learn_eval='eval'):
    # set the working directory and import helper functions
    # get the current working directory and then redirect into the functions under code
    cwd = os.getcwd()

    # import the results from the results folder
    results_cwd = os.path.join(os.path.abspath(cwd), 'results', 'decoupled-smoothing', learn_eval)

    predictions = {}
    file_path = os.path.join(results_cwd, method, data_nm, '{:04d}'.format(random_seed),
                             'inferred-predicates{:02d}'.format(int(pct_lbl*100)), 'GENDER.txt')
    with open(file_path, 'r') as f:
        for line in f:
            node, label, prob = line.strip().split('\t')
            node = int(node)
            label = int(label)
            prob = float(prob)
            if node in predictions:
                predictions[node][label] = prob
            else:
                predictions[node] = {label: prob}
    return predictions


def read_truth(data_nm, random_seed, pct_lbl, learn_eval='eval'):
    # set the working directory and import helper functions
    # get the current working directory and then redirect into the functions under code
    cwd = os.getcwd()

    # import the results from the results folder
    data_cwd = os.path.join(os.path.abspath(cwd), 'data', learn_eval, data_nm,
                               '{:02d}pct'.format(int(pct_lbl*100)), '{:04d}rand'.format(random_seed),
                               'gender_truth.txt')

    truth = {}
    with open(data_cwd, 'r') as f:
        for line in f:
            node, gender, true = line.strip().split('\t')
            if float(true) > 0:
                node = int(node)
                gender = int(gender)
                truth[node] = gender
    return truth


def find_tptn(predictions, truth):
    y_true = []
    y_score = []
    tp = []
    tp_score = []
    tn = []
    tn_score = []
    for node in predictions.keys():
        if node in truth:
            prob, predict = predictions[node]
            y_true.append([int(1 == truth[node]), int(2 == truth[node])])
            y_score.append([predictions[node][1], predictions[node][2]])

            tp.append(int(2 == truth[node]))
            tp_score.append(predictions[node][2])

            tn.append(int(1 == truth[node]))
            tn_score.append(predictions[node][1])
    return y_true, y_score, tp, tp_score, tn, tn_score


def calculate_metrics(truth, score):
    roc_score = roc_auc_score(truth, score, average="weighted")
    prc_score = average_precision_score(truth, score, average="weighted")
    return roc_score, prc_score

def main():
    # models = ['cli_decoupled_smoothing']
    models = ['cli_one_hop', 'cli_decoupled_smoothing_mod',
              'cli_decoupled_smoothing_prior', 'cli_decoupled_smoothing_partial']
    models = ['cli_two_hop']
    pct_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    random_seeds = [1, 12345, 837, 2841, 4293, 6305, 6746, 9056, 9241, 9547]
    results = {}

    for pct in pct_list:
        results['{:02d}pct'.format(int(pct*100))] = {}
        for model in models:
            results['{:02d}pct'.format(int(pct*100))][model] = {}
            for seed in random_seeds:
                print('model', model, 'pct', pct, 'random', seed)
                predictions = read_predictions(model, 'Amherst41', seed, pct)
                truth = read_truth('Amherst41', seed, pct)
                y_true, y_score, tp, tp_score, tn, tn_score = find_tptn(predictions, truth)

                roc_score, prc_score = calculate_metrics(tp, tp_score)
                results['{:02d}pct'.format(int(100 * pct))][model][seed] = {}
                results['{:02d}pct'.format(int(100*pct))][model][seed]['tp_roc'] = roc_score
                results['{:02d}pct'.format(int(100 * pct))][model][seed]['tp_prc'] = prc_score

                roc_score, prc_score = calculate_metrics(tn, tn_score)
                results['{:02d}pct'.format(int(100 * pct))][model][seed]['tn_roc'] = roc_score
                results['{:02d}pct'.format(int(100 * pct))][model][seed]['tn_prc'] = prc_score

                results['{:02d}pct'.format(int(100 * pct))][model][seed]['cat'] = np.mean(
                    np.equal(np.array(y_true).argmax(axis=-1), np.array(y_score).argmax(axis=-1)))

    # prc_file = "prc.csv"
    # roc_file = "roc.csv"
    # cat_file = "cat.csv"
    # with open(prc_file, 'w+') as f_prc, open(roc_file, 'w+') as f_roc, open(cat_file, 'w+') as f_cat:


    print(results)

main()