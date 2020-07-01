import os
from operator import add
from mpl_toolkits.axes_grid1 import Grid

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MultipleLocator
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


def draw_plots(pct_list, results, metric):
    """

    :param pct_list:
    :param results: an array composed of (mean, std, color) tuples
    :return:
    """

    metric_name = {'tp_roc': 'AUROC',
                   'tp_prc': 'AUPRC',
                   'tn_roc': 'TN AUROC',
                   'tn_prc': 'TN AUPRC',
                   'cat': 'Categorical Accuracy'}

    fig = plt.figure()
    # seaborn.set_style(style='white')
    grid = Grid(fig, rect=111, nrows_ncols=(1, 1),
                axes_pad=0.1, label_mode='L')
    for i in range(4):
        if i == 0:
            grid[i].xaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
            grid[i].yaxis.set_major_locator(FixedLocator([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))

            for mean, std, color in results:
                grid[i].errorbar(np.array(pct_list) * 100, mean,
                                 yerr=std, fmt='--o', capthick=2,
                                 alpha=1, elinewidth=3, color=color)

            """base_one_hop_mean = np.array([0.5242803169, 0.5646324657, 0.5860555728, 0.6133426532, 0.6258339368, 0.6440658034, 0.6507036668, 0.6549736731, 0.6693202023, 0.6510137064, 0.6435331092, 0.6651071383])
            base_one_hop_std = [0.02906211594,0.01851168638,0.01422862363,0.01377027519,0.01392284084,0.008649640963,0.007214889053,0.00704592724,0.01201646279,0.01508398614,0.03187498349,0.02802471537]
            base_one_hop_std = np.array(base_one_hop_std)
            base_two_hop_mean = [0.5809506394, 0.6451184766, 0.659092705, 0.7055699709, 0.7135068876, 0.7470305232, 0.7490983879, 0.7572306715, 0.7728814208, 0.7581824806, 0.7443194033, 0.7676921522]
            base_two_hop_mean = np.array(base_two_hop_mean)
            base_two_hop_std =[0.03824478186,0.01343521986,0.03107824273,0.02909771389,0.0195708889,0.04133852189,0.02497754726,0.02259123295,0.02304141732,0.03251789379,0.03765114997,0.03320440346]
            base_two_hop_std = np.array(base_two_hop_std)"""
            base_DS_mean = [0.5690678541, 0.6091408302, 0.6281314399, 0.6629018751, 0.6749368814,
                            0.7182350295, 0.724052967, 0.737606846, 0.7582979352, 0.7496986652,
                            0.7400014331, 0.7656347085]
            base_DS_mean = [0.5679863518, 0.6073961742, 0.6330774912, 0.6618386846, 0.6923111212,
                            0.7192616155, 0.7256418922, 0.7415268462, 0.7575042852, 0.7453494004,
                            0.7320081127, 0.7483847573]
            base_DS_mean = [0.5006461233, 0.5121698602, 0.5329688354, 0.5681426814, 0.5882642305,
                            0.5968852459, 0.6178149606, 0.6346863469, 0.638852459, 0.6572481572,
                            0.6294117647, 0.6607843137]
            base_DS_mean = np.array(base_DS_mean)
            base_ds_std = [0.04065870296, 0.01962305861, 0.0346181875, 0.03267599611, 0.02015367394,
                           0.04764108593, 0.02793457346, 0.02635875476, 0.02434629227,
                           0.03310654927, 0.03851662741, 0.03366887519]
            base_ds_std = [0.03901035364, 0.01959688957, 0.03018320853, 0.03644690394,
                           0.02024156679, 0.04314010932, 0.02671629424, 0.02880212332,
                           0.02742680543, 0.02833386157, 0.04996745409, 0.04470864294]
            base_ds_std = [0.001277829039, 0.02441892421, 0.0490220683, 0.0311144491, 0.02869807473,
                           0.0444755035, 0.031622072, 0.01652067246, 0.02013192181, 0.03335103676,
                           0.02791966836, 0.02409452103]
            base_ds_std = np.array(base_ds_std)

            """grid[i].errorbar(np.array(pct_list) * 100, base_one_hop_mean,
                                 yerr=base_one_hop_std, fmt='-o', capthick=2,
                                 alpha=1, elinewidth=3, color='orange')
            grid[i].errorbar(np.array(pct_list) * 100, base_two_hop_mean,
                                 yerr=base_two_hop_std, fmt='-o', capthick=2,
                                 alpha=1, elinewidth=3, color='darkcyan')"""
            grid[i].errorbar(np.array(pct_list) * 100, base_DS_mean,
                             yerr=base_ds_std, fmt='--o', capthick=2,
                             alpha=1, elinewidth=3, color='blue')

            random_mean = [0.5034257257, 0.5002619395, 0.4924728825, 0.5054212701, 0.5027953562,
                           0.5082933958, 0.4978260384, 0.4932371057, 0.5082893888, 0.4947455572,
                           0.5116322005, 0.4921932272]
            random_std = [0.009595442741, 0.01104561767, 0.01306187392, 0.02052901859,
                          0.01332757253, 0.0144329462, 0.01798360144, 0.01413479932, 0.01470998552,
                          0.03202642701, 0.0465864864, 0.03734775743]
            grid[i].errorbar(np.array(pct_list) * 100, random_mean,
                             yerr=random_std, fmt='-o', capthick=2,
                             alpha=1, elinewidth=3, color='green')
            top = 1
            # grid[i].annotate('ds-pref friend count', xy=(3, 0.68),
            #                color='green', alpha=1, size=12)
            grid[i].annotate('1-hop MV', xy=(3, top - .02),
                             color=(57 / 255, 108 / 255, 177 / 255), alpha=1, size=12)
            grid[i].annotate('2-hop MV', xy=(3, top - .06),
                             color='darkcyan', alpha=1, size=12)
            grid[i].annotate('random guessing', xy=(3, top - .1),
                             color='green', alpha=1, size=12)
            grid[i].annotate('decoupled smoothing', xy=(40, top - .02),
                             color='cornflowerblue', alpha=1, size=12)
            """grid[i].annotate('ds-prior', xy=(3, 0.84),
                             color='gold', alpha=1, size=12)
            grid[i].annotate('ds-partial', xy=(3, 0.80),
                             color='indianred', alpha=1, size=12)"""
            grid[i].annotate('ds preference concentration (200)', xy=(40, top - .06),
                             color='magenta', alpha=1, size=12)
            grid[i].annotate('ds preference concentration (normalized)', xy=(40, top - .1),
                             color='purple', alpha=1, size=12)
            """grid[i].annotate('ds-closerfriend (5)', xy=(50, 0.88),
                             color='purple', alpha=1, size=12)"""
            # grid[i].annotate('ds-pref cf threshold', xy=(3, 0.92),
            #                 color='black', alpha=1, size=12)
            # grid[i].annotate('ds-pref homophily', xy=(3, 0.92),
            #                 color='olivedrab', alpha=1, size=12)
            # grid[i].annotate('ds-pref cluster coeff', xy=(3, 0.96),
            #                 color='purple', alpha=1, size=12)

            grid[i].set_ylim(0.4, 1)
            grid[i].set_xlim(0, 100)

            grid[i].spines['right'].set_visible(False)
            grid[i].spines['top'].set_visible(False)
            grid[i].tick_params(axis='both', which='major', labelsize=13)
            grid[i].tick_params(axis='both', which='minor', labelsize=13)
            grid[i].set_xlabel('Percent of Nodes Initially Labeled').set_fontsize(15)
            grid[i].set_ylabel(metric_name[metric]).set_fontsize(15)

    grid[0].set_xticks([0, 25, 50, 75, 100])
    grid[0].set_yticks([0.4, 0.6, 0.8, 1])

    grid[0].minorticks_on()
    grid[0].tick_params('both', length=4, width=1, which='major', left=1, bottom=1, top=0, right=0)

    # plt.show()
    plt.legend()
    plt.savefig('PSL-DS {} Figure'.format(metric_name[metric]))
    fig.clf()


def draw_plots2(pct_list, results, metric):
    fig, ax = plt.subplots()
    npct_list = np.array(pct_list) * 100

    # title, labels
    ax.set_title('PSL-DS {} Scores'.format(metric), fontsize=20)
    ax.set_xlabel('Percent of Nodes Initially Labeled').set_fontsize(15)
    ax.set_ylabel(metric).set_fontsize(15)

    # fix ticks on x axis
    plt.xlim((0,100.1))
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    # fix ticks on y axis
    plt.ylim((0.45, 0.8))
    # FixedLocator([0, 25, 50, 75, 100])
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))

    # only leave bottom-left spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    base_ds_mean = []
    base_ds_std = []
    random_mean = []
    random_std = []
    if metric == 'AUROC':
        base_ds_mean = [0.5690678541, 0.6091408302, 0.6281314399, 0.6629018751, 0.6749368814,
                        0.7182350295, 0.724052967, 0.737606846, 0.7582979352, 0.7496986652,
                        0.7400014331, 0.7656347085]
        base_ds_std = [0.04065870296, 0.01962305861, 0.0346181875, 0.03267599611, 0.02015367394,
                       0.04764108593, 0.02793457346, 0.02635875476, 0.02434629227,
                       0.03310654927, 0.03851662741, 0.03366887519]
        random_mean = [0.5034257257, 0.5002619395, 0.4924728825, 0.5054212701, 0.5027953562,
                       0.5082933958, 0.4978260384, 0.4932371057, 0.5082893888, 0.4947455572,
                       0.5116322005, 0.4921932272]
        random_std = [0.009595442741, 0.01104561767, 0.01306187392, 0.02052901859,
                      0.01332757253, 0.0144329462, 0.01798360144, 0.01413479932, 0.01470998552,
                      0.03202642701, 0.0465864864, 0.03734775743]
    elif metric == 'AUPRC':
        base_ds_mean = [0.5679863518, 0.6073961742, 0.6330774912, 0.6618386846, 0.6923111212,
                        0.7192616155, 0.7256418922, 0.7415268462, 0.7575042852, 0.7453494004,
                        0.7320081127, 0.7483847573]
        base_ds_std = [0.03901035364, 0.01959688957, 0.03018320853, 0.03644690394,
                       0.02024156679, 0.04314010932, 0.02671629424, 0.02880212332,
                       0.02742680543, 0.02833386157, 0.04996745409, 0.04470864294]
        random_mean = [0.50206179, 0.5019224376, 0.4963246408, 0.5039353471, 0.5031874948,
                       0.5043250761, 0.4995631792, 0.4995437831, 0.506819437, 0.4960391298,
                       0.5091311614, 0.4729111382, 0.5485188145]
        random_std = [0.005505289106, 0.006111154255, 0.008806653489, 0.01384418448, 0.009641125972,
                      0.0128902026, 0.008250374928, 0.01698904403, 0.01879925757, 0.02969468559,
                      0.03436020741, 0.04561989572, 0.1190351711]
    elif metric == 'Categorical Accuracy':
        base_ds_mean = [0.5006461233, 0.5121698602, 0.5329688354, 0.5681426814, 0.5882642305,
                        0.5968852459, 0.6178149606, 0.6346863469, 0.638852459, 0.6572481572,
                        0.6294117647, 0.6607843137]
        base_ds_std = [0.001277829039, 0.02441892421, 0.0490220683, 0.0311144491, 0.02869807473,
                       0.0444755035, 0.031622072, 0.01652067246, 0.02013192181, 0.03335103676,
                       0.02791966836, 0.02409452103]
        random_mean = [0.5032306163, 0.4996374935, 0.4945872061, 0.5028905289, 0.5007027407,
                       0.5097540984, 0.4993110236, 0.4928659287, 0.5045901639, 0.5007371007,
                       0.5200980392, 0.4882352941, 0.5142857143]
        random_std = [0.009635398293, 0.01131643067, 0.01368736951, 0.02246816857, 0.01321472261,
                      0.01386433674, 0.01863476627, 0.01406080951, 0.01719011387, 0.02491701474,
                      0.0384907116, 0.03195269689, 0.1485632883]
    ax.errorbar(npct_list, base_ds_mean[:11], yerr=base_ds_std[:11], label='Org-DS',
                    fmt='--o', elinewidth=3, capthick=2, color='blueviolet')
    """ax.errorbar(npct_list, random_mean[:11], yerr=random_std[:11], label='Random Guess',
                fmt='--o', elinewidth=3, capthick=2, color='maroon')"""

    # draw the error bars
    for name, mean, std, color, format in results:
        ax.errorbar(npct_list, mean, yerr=std, label=name, fmt=format, elinewidth=3,
                    capthick=2, color=color)

    # add the legend
    figlegend = plt.figure()
    ax_leg = figlegend.add_subplot(111)
    legend = ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=6, frameon=False)
    ax_leg.axis('off')
    # plt.legend(loc="upper left", prop=dict(size=8))
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # plt.show()
    plt.savefig('PSL-DS {} Figure'.format(metric), figsize=(10, 10), dpi=100)
    figlegend.savefig('PSL-DS {} Legend'.format(metric),
                      bbox_inches=legend.get_window_extent().transformed(figlegend.dpi_scale_trans.inverted()))
    fig.clf()


def main():
    models = ['cli_one_hop', 'cli_two_hop', 'cli_decoupled_smoothing_mod_h2',
              'cli_decoupled_smoothing_prior', 'cli_decoupled_smoothing_partial',
              'cli_decoupled_smoothing_pref_homophily',
              'cli_decoupled_smoothing_closerfriend_threshold']
    """models = ['cli_decoupled_smoothing_prior', 'cli_decoupled_smoothing_mod',
              'cli_decoupled_smoothing_closerfriend_thres20_var',
              'cli_decoupled_smoothing_closerfriend_thres10_var',
              'cli_decoupled_smoothing_closerfriend_thres05_var',
              'cli_decoupled_smoothing_closerfriend_thres100_var',
              'cli_decoupled_smoothing_closerfriend_thres150_var',
              'cli_decoupled_smoothing_closerfriend_thres200_var',
              'cli_decoupled_smoothing_closefriend_normalized',
              'cli_decoupled_smoothing_closefriend']"""
    models = ['cli_two_hop', 'cli_one_hop', 'cli_oh_prior', 'cli_decoupled_smoothing_mod',
              'cli_ds_prior', 'cli_decoupled_smoothing_closefriend_normalized', 'cli_ds_norm_prior',
              'cli_decoupled_smoothing_closerfriend_thres200_var', 'cli_ds_t200_prior']
    metrics = ['tp_roc', 'tp_prc', 'tn_roc', 'tn_prc', 'cat']
    # pct_list = [0.01, 0.1, 0.3, 0.5, 0.8]
    pct_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # pct_list = [0.1, 0.3, 0.5, 0.8, 0.95]
    random_seeds = [1, 12345, 837, 2841, 4293, 6305, 6746, 9056, 9241, 9547]

    roc_prc_cat = {'cat': {}}
    for metric in metrics:
        roc_prc_cat[metric] = {}
    for seed in random_seeds:
        for metric in ['tp_roc', 'tp_prc', 'tn_roc', 'tn_prc', 'cat']:
            roc_prc_cat[metric][seed] = {}
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
            roc_prc_cat['tp_roc'][seed][model] = tp_roc
            roc_prc_cat['tp_prc'][seed][model] = tp_prc
            roc_prc_cat['tn_roc'][seed][model] = tn_roc
            roc_prc_cat['tn_prc'][seed][model] = tn_prc
            roc_prc_cat['cat'][seed][model] = cat

    for metric in ['tp_roc', 'tp_prc', 'tn_roc', 'tn_prc', 'cat']:
        with open('{}.csv'.format(metric), 'w+') as f:
            f.write('{},{}\n'.format(
                'model',
                ','.join([str(pct) for pct in pct_list])
            ))
            averaged_model_scores = {}
            for random in roc_prc_cat[metric].keys():
                f.write('{}\n'.format(random))
                for model, calc_pct in roc_prc_cat[metric][random].items():
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
                averaged_model_scores[model] = [i / len(random_seeds)
                                                for i in averaged_model_scores[model]]
                f.write('{},{}\n'.format(
                    model,
                    ','.join([str(pct) for pct in averaged_model_scores[model]])
                ))

    color_map = {'cli_one_hop': 'royalblue',
        'cli_oh_prior': 'royalblue',
                 'cli_one_hop_gpp': 'royalblue',
                 'cli_two_hop': 'orange',
                 'cli_decoupled_smoothing_mod': 'green',
                 'cli_decoupled_smoothing_closerfriend_thres200_var': 'red',
                 'cli_decoupled_smoothing_closefriend_normalized': 'gray',
                 'cli_decoupled_smoothing_closerfriend_thres05_var': 'magenta'}
    label_name = {'cli_one_hop': '1-hop',
                    'cli_oh_prior': '1-Hop',
                  'cli_one_hop_gpp': '1-Hop',
                  'cli_two_hop': '2-Hop',
                  'cli_decoupled_smoothing_mod': 'PSL-DS',
                  'cli_decoupled_smoothing_closerfriend_thres200_var': 'DS Pref Con (200)',
                  'cli_decoupled_smoothing_closefriend_normalized': 'DS Pref Con (normalized)',
                  'cli_decoupled_smoothing_closerfriend_thres05_var': 'magenta'}
    format_guide = {'cli_one_hop': '--o',
                    'cli_oh_prior': '--o',
                  'cli_one_hop_gpp': '--o',
                  'cli_two_hop': '--o',
                  'cli_decoupled_smoothing_mod': '-o',
                  'cli_decoupled_smoothing_closerfriend_thres200_var': '-o',
                  'cli_decoupled_smoothing_closefriend_normalized': '-o',
                  'cli_decoupled_smoothing_closerfriend_thres05_var': '-o'}

    metric_name = {'tp_roc': 'AUROC',
                   'tp_prc': 'AUPRC',
                   'tn_roc': 'TN AUROC',
                   'tn_prc': 'TN AUPRC',
                   'cat': 'Categorical Accuracy'}

    """for metric in ['tp_roc', 'cat']:
        results = []
        for model in models:
            temp_array = np.array([roc_prc_cat[metric][seed][model] for seed in random_seeds])
            mean_values = list(temp_array.mean(axis=0))
            std_values = list(temp_array.std(axis=0))

            results.append((label_name[model], mean_values, std_values, color_map[model], format_guide[model]))

        draw_plots2(pct_list, results, metric_name[metric])
        # draw_plots(pct_list, results, metric)"""


main()
