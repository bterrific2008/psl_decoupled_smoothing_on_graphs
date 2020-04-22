from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot as plt
import random as rand
import numpy as np

import csv

predictions = {}

auroc_scores = {}
categorical_accuracy = {}
auprc_scores = {}

debug = False

# models = ['cli_decoupled_smoothing']
models = ['cli_one_hop', 'cli_two_hop', 'cli_decoupled_smoothing_mod',
          'cli_decoupled_smoothing_partial', 'cli_decoupled_smoothing_prior']
pct_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

random_seed = [1, 12345, 837, 2841, 4293, 6305, 6746, 9056, 9241, 9547][9]

output_file = 'results{:04d}.csv'.format(random_seed)

for model in models:
    for pct in pct_list:
        with open('results/decoupled-smoothing/{0}/Amherst41/{1:04d}/inferred'
                  '-predicates{2:02d}/GENDER.txt'.format(model, random_seed, int(pct * 100)),
                  'r') as f:
            for line in f:
                node, label, prob = line.strip().split('\t')
                node = int(node)
                label = int(label)
                prob = float(prob)
                if node in predictions:
                    predictions[node][label] = prob
                else:
                    predictions[node] = {label: prob}
        truth = {}
        with open('data/Amherst41/{:02d}pct/{:04d}rand/gender_truth.txt'.format(int(pct * 100),
                                                                            random_seed),
                  'r') as f:
            for line in f:
                node, gender, true = line.strip().split('\t')
                if float(true) > 0:
                    node = int(node)
                    gender = int(gender)
                    truth[node] = gender
        y_true = []
        y_score = []

        tp = []
        tn = []
        tp_score = []
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

        if model in categorical_accuracy:
            categorical_accuracy[model].append(np.mean(
                np.equal(np.array(y_true).argmax(axis=-1), np.array(y_score).argmax(axis=-1))))
        else:
            categorical_accuracy[model] = [np.mean(
                np.equal(np.array(y_true).argmax(axis=-1), np.array(y_score).argmax(axis=-1)))]

        calculated_score = roc_auc_score(tp, tp_score, average="weighted")
        if debug:
            print('tp {} {} Weighted AUCROC Score'.format(model, pct), calculated_score)
        if 'tp{}'.format(model) in auroc_scores:
            auroc_scores['tp{}'.format(model)].append(calculated_score)
        else:
            auroc_scores['tp{}'.format(model)] = [calculated_score]

        calculated_score = average_precision_score(tp, tp_score, average="weighted")
        if debug:
            print('tp {} {} Weighted AUCPRC Score'.format(model, pct), calculated_score)
        if 'tp{}'.format(model) in auprc_scores:
            auprc_scores['tp{}'.format(model)].append(calculated_score)
        else:
            auprc_scores['tp{}'.format(model)] = [calculated_score]

        calculated_score = roc_auc_score(tn, tn_score, average="weighted")
        if debug:
            print('tn{} {} Weighted AUCROC Score'.format(model, pct), calculated_score)
        if 'tn{}'.format(model) in auroc_scores:
            auroc_scores['tn{}'.format(model)].append(calculated_score)
        else:
            auroc_scores['tn{}'.format(model)] = [calculated_score]

        calculated_score = average_precision_score(tn, tn_score, average="weighted")
        if debug:
            print('tn{} {} Weighted AUCPRC Score'.format(model, pct), calculated_score)
        if 'tn{}'.format(model) in auprc_scores:
            auprc_scores['tn{}'.format(model)].append(calculated_score)
        else:
            auprc_scores['tn{}'.format(model)] = [calculated_score]

### random

models.append('random')
for pct in pct_list:
    with open('data/Amherst41/{:02d}pct/{:04d}rand/gender_test_indicies.txt'
                      .format(int(pct * 100), random_seed), 'r') as f:
        for line in f:
            node = line.strip().split('\t')[0]
            node = int(node)
            if node in predictions:
                choices = [1, 2]
                rand.shuffle(choices)
                predictions[node][choices[0]] = 1
                predictions[node][choices[1]] = 0
    truth = {}
    with open(
            'data/Amherst41/{:02d}pct/{:04d}rand/gender_truth.txt'.format(int(pct * 100), random_seed),
            'r') as f:
        for line in f:
            node, gender, true = line.strip().split('\t')
            if float(true) > 0:
                node = int(node)
                gender = int(gender)
                truth[node] = gender
    y_true = []
    y_score = []

    tp = []
    tn = []
    tp_score = []
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

    model = 'random'
    if model in categorical_accuracy:
        categorical_accuracy[model].append(
            np.mean(np.equal(np.array(y_true).argmax(axis=-1), np.array(y_score).argmax(axis=-1))))
    else:
        categorical_accuracy[model] = [
            np.mean(np.equal(np.array(y_true).argmax(axis=-1), np.array(y_score).argmax(axis=-1)))]

    calculated_score = roc_auc_score(tp, tp_score, average="weighted")
    if debug:
        print('tp {} {} Weighted AUCROC Score'.format(model, pct), calculated_score)
    if 'tp{}'.format(model) in auroc_scores:
        auroc_scores['tp{}'.format(model)].append(calculated_score)
    else:
        auroc_scores['tp{}'.format(model)] = [calculated_score]

    calculated_score = average_precision_score(tp, tp_score, average="weighted")
    if debug:
        print('tp {} {} Weighted AUCPRC Score'.format(model, pct), calculated_score)
    if 'tp{}'.format(model) in auprc_scores:
        auprc_scores['tp{}'.format(model)].append(calculated_score)
    else:
        auprc_scores['tp{}'.format(model)] = [calculated_score]

    calculated_score = roc_auc_score(tp, tp_score, average="weighted")
    if debug:
        print('tn {} {} Weighted AUCROC Score'.format(model, pct), calculated_score)
    if 'tn{}'.format(model) in auroc_scores:
        auroc_scores['tn{}'.format(model)].append(calculated_score)
    else:
        auroc_scores['tn{}'.format(model)] = [calculated_score]

    calculated_score = average_precision_score(tp, tp_score, average="weighted")
    if debug:
        print('tn {} {} Weighted AUCPRC Score'.format(model, pct), calculated_score)
    if 'tn{}'.format(model) in auprc_scores:
        auprc_scores['tn{}'.format(model)].append(calculated_score)
    else:
        auprc_scores['tn{}'.format(model)] = [calculated_score]

# fig, ax = plt.subplots()
# for model in auroc_scores.keys():
#     ax.plot(pct_list, auroc_scores[model])
# ax.set(xlabel='Percent Labeled', ylabel='AUROC', title='AUROC')
# ax.grid()
# plt.legend(auroc_scores.keys(), loc='upper left')
# fig.savefig('results.png')
# plt.show()

with open(output_file, mode='w+') as f:

    string_pct = ",".join(["{:02}%".format(int(i * 100)) for i in pct_list])
    f.write('Model{},{}\n'.format(random_seed, string_pct))

    f.write('AUROC\n')
    for key in auroc_scores.keys():
        f.write("{},{}\n".format(key, ",".join([str(i) for i in auroc_scores[key]])))

    f.write('AUPRC\n')
    for key in auprc_scores.keys():
        f.write("{},{}\n".format(key, ",".join([str(i) for i in auprc_scores[key]])))

    f.write('Categorical Accuracy\n')
    for key in categorical_accuracy.keys():
        f.write("{},{}\n".format(key, ",".join([str(i) for i in categorical_accuracy[key]])))
