import argparse
import datetime
import os
import random
from pathlib import Path

import parsing as parse_mat


def generate_data(percent_labeled, random_seed=1, school_data='Amherst41.mat'):
    # parse the data
    adj_matrix, gender_unknown, gender_y = parse_data(school_data)

    # write the data
    if isinstance(percent_labeled, list):
        for pct_label in percent_labeled:
            write_files(adj_matrix, gender_unknown, gender_y, random_seed, pct_label,
                        school_data)
    else:
        write_files(adj_matrix, gender_unknown, gender_y, random_seed, percent_labeled, school_data)


def parse_data(school_data='Amherst41.mat'):
    # set the working directory and import helper functions
    # get the current working directory and then redirect into the functions under code
    cwd = os.getcwd()

    # import the data from the data folder
    data_cwd = '{0}\\{1}'.format(os.path.abspath(cwd), 'data')
    fb100_file = '{0}\\{1}'.format(data_cwd, school_data)
    A, metadata = parse_mat.parse_fb100_mat_file(fb100_file)

    # change A(scipy csc matrix) into a numpy matrix
    adj_matrix = A.todense().tolist()
    # get the gender for each node (1/2, 0 for missing)
    gender_y_tmp = metadata[:, 1]
    gender_unknown = []
    gender_y = []
    for i, y in enumerate(gender_y_tmp):
        if y > 0:
            gender_y.append((i, y))
        else:
            gender_unknown.append(i)

    return adj_matrix, gender_unknown, gender_y


def write_files(adj_matrix, gender_unknown, gender_y, random_seed=1, percent_labeled=0.01,
                data_name='Amherst41'):
    # fix random
    random.seed(random_seed)

    # set the working directory and import helper functions
    # get the current working directory and then redirect into the functions under code
    cwd = os.getcwd()

    # import the data from the data folder
    data_cwd = '{0}\\{1}'.format(os.path.abspath(cwd), 'data')

    # new directory for school
    data_cwd = '{0}\\{1}'.format(data_cwd, data_name.split('.')[0])
    Path(data_cwd).mkdir(parents=True, exist_ok=True)  # if directory doesn't exist, create it

    # new directory for % labeled
    data_cwd = '{0}\\{1:02d}'.format(data_cwd, int(percent_labeled * 100))
    Path(data_cwd).mkdir(parents=True, exist_ok=True)  # if directory doesn't exist, create it

    # collect the edges
    edges = []
    preferences_file = data_cwd + "\\preferences_targets.txt"
    with open(preferences_file, 'w+') as f:
        for row_i, row in enumerate(adj_matrix):
            f.write('{0}\t1\n'.format(row_i))
            f.write('{0}\t2\n'.format(row_i))
            for col_i, column in enumerate(row):
                if column > 0:
                    edges.append((row_i, col_i))

    # write out the edge data
    edges_file = data_cwd + '\\edges_obs'
    with open(edges_file + '.txt', 'w+') as f:
        for edge in edges:
            f.write('{0[0]}\t{0[1]}\t1.0\n'.format(edge))

    gender_truth_file = data_cwd + '\\gender_truth.txt'
    gender_obs_file = data_cwd + '\\gender_obs.txt'
    gender_targets_file = data_cwd + '\\gender_targets.txt'

    with open(gender_obs_file, 'w+') as f_obs, open(gender_targets_file, 'w+') as f_target, open(
            gender_truth_file, 'w+') as f_truth:

        gender2 = gender_y.copy()
        random.shuffle(gender2)
        split = int(len(gender2) * percent_labeled)
        for gender_i, gender in gender2[:split]:
            if gender > 0:
                f_obs.write('{0}\t{1}\t{2}\n'.format(gender_i, 1, float(gender == 1)))
                f_obs.write('{0}\t{1}\t{2}\n'.format(gender_i, 2, float(gender == 2)))
        for gender_i, gender in gender2[split:]:
            if gender > 0:
                f_target.write('{0}\t1\n'.format(gender_i))
                f_target.write('{0}\t2\n'.format(gender_i))
                f_truth.write('{0}\t{1}\t{2}\n'.format(gender_i, 1, float(gender == 1)))
                f_truth.write('{0}\t{1}\t{2}\n'.format(gender_i, 2, float(gender == 2)))

        for gender_i in gender_unknown:
            f_target.write('{0}\t1\n'.format(gender_i))
            f_target.write('{0}\t2\n'.format(gender_i))

    data_log = data_cwd + '\\data_log.txt'
    with open(data_log, 'w+') as f:
        f.write('Timestamp\t{}\n'.format(datetime.datetime.now()))
        f.write('Data used\t{}\n'.format(data_name))
        f.write('% Labeled\t{}\n'.format(percent_labeled))
        f.write('Random Seed\t{}\n'.format(random_seed))


cli_parse = argparse.ArgumentParser()
cli_parse.add_argument("--seed", help="Sets a random seed", default=1)
cli_parse.add_argument("--pct", help="Sets the % of labeled data", default=.5)
cli_parse.add_argument("--data", help="Specifies the name of the data file to use",
                       default='Amherst41.mat')
args = cli_parse.parse_args()

# generate_data(args.pct, args.seed, args.data)
generate_data([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
