from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

predictions = {}
scores = {}
models = ['cli_one_hop', 'cli_two_hop', 'cli_decoupled_smoothing_mod']
pct_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
for model in models:
    for pct in pct_list:
        with open('results/decoupled-smoothing/{0}/Amherst41/1/inferred'
                  '-predicates{1:02d}/GENDER.txt'.format(model, int(pct * 100)), 'r') as f:
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
        with open('data/Amherst41/rand1/{:02d}pct/gender_truth.txt'.format(int(pct * 100)),
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

        calculated_score = roc_auc_score(tp, tp_score, average="weighted")
        print('tp {} {} Weighted AUCROC Score'.format(model, pct), calculated_score)
        if 'tp{}'.format(model) in scores:
            scores['tp{}'.format(model)].append(calculated_score)
        else:
            scores['tp{}'.format(model)] = [calculated_score]

        calculated_score = roc_auc_score(tn, tn_score, average="weighted")
        print('{} {} Weighted AUCROC Score'.format(model, pct), calculated_score)
        if 'tn{}'.format(model) in scores:
            scores['tn{}'.format(model)].append(calculated_score)
        else:
            scores['tn{}'.format(model)] = [calculated_score]

fig, ax = plt.subplots()
for model in scores.keys():
    ax.plot(pct_list, scores[model])
ax.set(xlabel='Percent Labeled', ylabel='AUROC',title='AUROC')
ax.grid()
plt.legend(scores.keys(), loc='upper left')
fig.savefig('results.png')
plt.show()
