import os, re
import matplotlib.pyplot as plt

# set the working directory and import helper functions
# get the current working directory and then redirect into the functions under code
cwd = os.getcwd()
results_dir = os.path.join(cwd, 'decoupled-smoothing', 'cli_one_hop', 'Amherst41', '1')

auroc = []
pos_auprc = []
neg_auprc = []
cat_acc = []
pct = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

for pct_text in ['01', '05', '10', '20', '30', '40', '50', '60', '70', '80', '90', '95', '99']:
    output_file = os.path.join(results_dir, 'out{}.txt'.format(pct_text))
    # utput_file = output_file.replace('\\', '/')
    with open(output_file, 'r') as f:
        for line in f:
            cat_acc_temp = re.search('(?<=Categorical Accuracy: )(0\.[\d]+)', line)
            if cat_acc_temp:
                cat_acc.append(cat_acc_temp)

            auroc_temp = re.search('(?<=AUROC: )(0\.[\d]+)', line)
            if auroc_temp:
                auroc.append(float(auroc_temp.group(0)))
                print(auroc_temp.group(0))
            pos_auprc_temp = re.search('(?<=Positive Class AUPRC: )(0\.[\d]+)', line)
            if pos_auprc_temp:
                pos_auprc.append(pos_auprc_temp)
                print(pos_auprc_temp.group(0))
            neg_auprc_temp = re.search('(?<=Negative Class AUPRC: )(0\.[\d]+)', line)
            if neg_auprc_temp:
                neg_auprc.append(neg_auprc_temp)
                print(neg_auprc_temp.group(0))

print(len(pct), len(auroc))
fig, ax = plt.subplots()
ax.plot(pct, auroc)

results_dir = os.path.join(cwd, 'decoupled-smoothing', 'cli_two_hop', 'Amherst41', '1')

auroc = []
pos_auprc = []
neg_auprc = []
cat_acc = []
pct = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

for pct_text in ['01', '05', '10', '20', '30', '40', '50', '60', '70', '80', '90', '95', '99']:
    output_file = os.path.join(results_dir, 'out{}.txt'.format(pct_text))
    # utput_file = output_file.replace('\\', '/')
    with open(output_file, 'r') as f:
        for line in f:
            cat_acc_temp = re.search('(?<=Categorical Accuracy: )(0\.[\d]+)', line)
            if cat_acc_temp:
                cat_acc.append(cat_acc_temp)

            auroc_temp = re.search('(?<=AUROC: )(0\.[\d]+)', line)
            if auroc_temp:
                auroc.append(float(auroc_temp.group(0)))
                print(auroc_temp.group(0))
            pos_auprc_temp = re.search('(?<=Positive Class AUPRC: )(0\.[\d]+)', line)
            if pos_auprc_temp:
                pos_auprc.append(pos_auprc_temp)
                print(pos_auprc_temp.group(0))
            neg_auprc_temp = re.search('(?<=Negative Class AUPRC: )(0\.[\d]+)', line)
            if neg_auprc_temp:
                neg_auprc.append(neg_auprc_temp)
                print(neg_auprc_temp.group(0))

print(pct)
print(auroc)
ax.plot(pct, auroc)

results_dir = os.path.join(cwd, 'decoupled-smoothing', 'cli_decoupled_smoothing', 'Amherst41', '1')

auroc = []
pos_auprc = []
neg_auprc = []
cat_acc = []
pct = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

for pct_text in ['01', '05', '10', '20', '30', '40', '50', '60', '70', '80', '90', '95', '99']:
    output_file = os.path.join(results_dir, 'out{}.txt'.format(pct_text))
    # utput_file = output_file.replace('\\', '/')
    with open(output_file, 'r') as f:
        for line in f:
            cat_acc_temp = re.search('(?<=Categorical Accuracy: )(0\.[\d]+)', line)
            if cat_acc_temp:
                cat_acc.append(cat_acc_temp)

            auroc_temp = re.search('(?<=AUROC: )(0\.[\d]+)', line)
            if auroc_temp:
                auroc.append(float(auroc_temp.group(0)))
                print(auroc_temp.group(0))
            pos_auprc_temp = re.search('(?<=Positive Class AUPRC: )(0\.[\d]+)', line)
            if pos_auprc_temp:
                pos_auprc.append(pos_auprc_temp)
                print(pos_auprc_temp.group(0))
            neg_auprc_temp = re.search('(?<=Negative Class AUPRC: )(0\.[\d]+)', line)
            if neg_auprc_temp:
                neg_auprc.append(neg_auprc_temp)
                print(neg_auprc_temp.group(0))

print(pct)
print(auroc)
ax.plot(pct, auroc)

ax.set(xlabel='Percent Labeled', ylabel='AUROC',title='AUROC')
ax.grid()

plt.legend(['one hop', 'two hop', 'decoupled smoothing'], loc='upper left')
fig.savefig('test.png')
plt.show()
