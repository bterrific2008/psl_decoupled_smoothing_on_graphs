from sklearn.metrics import roc_curve, auc

predictions = []
with open('cli_one_hop/inferred-predicates/GENDER.txt', 'r') as f:
    for line in f:
        node, prediction = line.strip().split('\t')
        predictions.append((node, prediction))
print(predictions)
truth = {}
with open('data/gender_truth.txt', 'r') as f:
    for line in f:
        node, gender = line.strip().split('\t')
        gender = float(gender)
        truth[node] = gender
print('test')
y_true = []
y_score = []
for node, score in predictions:
    if node in truth:
        y_true.append(float(truth[node]))
        y_score.append(float(score))
fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
print('AUC score', auc(fpr, tpr))