import os
from operator import add

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def read_predictions(method, data_nm, random_seed, pct_lbl, learn_eval='eval'):
    # set the working directory and import helper functions
    # get the current working directory and then redirect into the functions under code
    cwd = os.getcwd()

    # import the results from the results folder
    results_cwd = os.path.join(os.path.abspath(cwd), 'results', 'decoupled-smoothing', learn_eval)

    # read the predictions data
    predictions = {}
    file_path = os.path.join(results_cwd, method, data_nm, '{:04d}'.format(random_seed),
                             'inferred-predicates{:02d}'.format(int(pct_lbl * 100)), 'GENDER.txt')
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
                            '{:02d}pct'.format(int(pct_lbl * 100)),
                            '{:04d}rand'.format(random_seed),
                            'gender_truth.txt')

    # read the truth data
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
            # find truth data per node, and associated score
            y_true.append([int(1 == truth[node]), int(2 == truth[node])])
            y_score.append([predictions[node][1], predictions[node][2]])

            # find true positives
            tp.append(int(2 == truth[node]))
            tp_score.append(predictions[node][2])

            # find true negatives
            tn.append(int(1 == truth[node]))
            tn_score.append(predictions[node][1])

    return y_true, y_score, tp, tp_score, tn, tn_score


# find roc and prc scores
def calculate_metrics(truth, score):
    roc_score = roc_auc_score(truth, score, average="weighted")
    prc_score = average_precision_score(truth, score, average="weighted")
    return roc_score, prc_score


def main():
    # models = ['cli_one_hop', 'cli_decoupled_smoothing_mod',
    #           'cli_decoupled_smoothing_prior', 'cli_decoupled_smoothing_partial']
    models = ['cli_combo']
    pct_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    random_seeds = [1, 12345, 837, 2841, 4293, 6305, 6746, 9056, 9241, 9547]

    results = {'roc': {}, 'prc': {}, 'cat': {}}
    for metric in ['tp_roc', 'tp_prc', 'tn_roc', 'tn_prc', 'cat']:
        results[metric] = {}
    for seed in random_seeds:
        for metric in ['tp_roc', 'tp_prc', 'tn_roc', 'tn_prc', 'cat']:
            results[metric][seed] = {}
        for model in models:
            tp_roc = []
            tp_prc = []
            tn_roc = []
            tn_prc = []
            cat = []
            for pct in pct_list:
                predictions = read_predictions(model, 'Amherst41', seed, pct)
                truth = read_truth('Amherst41', seed, pct)
                y_true, y_score, tp, tp_score, tn, tn_score = find_tptn(predictions, truth)

                roc_score, prc_score = calculate_metrics(tp, tp_score)
                tp_roc.append(roc_score)
                tp_prc.append(prc_score)

                roc_score, prc_score = calculate_metrics(tn, tn_score)
                tn_roc.append(roc_score)
                tn_prc.append(prc_score)

                cat.append(np.mean(
                    np.equal(np.array(y_true).argmax(axis=-1), np.array(y_score).argmax(axis=-1))))
            results['tp_roc'][seed][model] = tp_roc
            results['tp_prc'][seed][model] = tp_prc
            results['tn_roc'][seed][model] = tn_roc
            results['tn_prc'][seed][model] = tn_prc
            results['cat'][seed][model] = cat

    for metric in ['tp_roc', 'tp_prc', 'tn_roc', 'tn_prc', 'cat']:
        with open('{}.csv'.format(metric), 'w+') as f:
            f.write('{},{}\n'.format(
                'model',
                ','.join([str(pct) for pct in pct_list])
            ))
            averaged_model_scores = {}
            for random in results[metric].keys():
                f.write('{}\n'.format(random))
                for model, calc_pct in results[metric][random].items():
                    f.write('{},{}\n'.format(
                        model,
                        ','.join([str(pct) for pct in calc_pct])))
                    if model not in averaged_model_scores:
                        averaged_model_scores[model] = calc_pct
                    else:
                        averaged_model_scores[model] = list(map(add, averaged_model_scores[model],
                                                                calc_pct))
            f.write('average\n')
            for model in averaged_model_scores.keys():
                averaged_model_scores[model] = [i/len(random_seeds)
                                              for i in averaged_model_scores[model]]
                f.write('{},{}\n'.format(
                    model,
                    ','.join([str(pct) for pct in averaged_model_scores[model]])
                ))

main()
