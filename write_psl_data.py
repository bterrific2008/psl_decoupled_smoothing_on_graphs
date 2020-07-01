import argparse
import datetime
import os
import random
import json
import networkx as nx
from pathlib import Path

import parsing as parse_mat


def generate_data(random_seed=1, school_data='Amherst41.mat', learn=False):
    """
    Generates data

    :param random_seed: (str) random seed used to generate the data
    :param school_data: (str) filename containing the school data
    :param learn: (bool) specifies if the data will be used for learning or evaluation

    :return: nothing
    """

    # fix random
    random.seed(random_seed)

    # parse the data
    adj_matrix, gender_y = parse_data(school_data)

    # write the data
    for pct_label in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        print('Creating data for {} at {}% labeled for {} with random seed'.format(school_data,
                                                                                   pct_label,
                                                                                   "learning" if learn
                                                                                   else "evaluation",
                                                                                   random_seed))
        write_files(adj_matrix, gender_y, random_seed, pct_label,
                    school_data, learn)


def parse_data(school_data='Amherst41.mat'):
    """
    Converts the mat file

    :param school_data: (str) filename containing the school data
    :param learn: (bool) specifies if the data will be used for learning or evaluation

    :return: adj_matrix (list),  gender_y (list)
    """

    # set the working directory and import helper functions
    # get the current working directory and then redirect into the functions under code
    cwd = os.getcwd()

    # import the data from the data folder
    data_cwd = '{0}/{1}'.format(os.path.abspath(cwd), 'data')
    fb100_file = '{0}/{1}'.format(data_cwd, school_data)
    A, metadata = parse_mat.parse_fb100_mat_file(fb100_file)

    ## function to create + save dictionary of features
    def create_dict(key, obj):
        return (dict([(key[i], obj[i]) for i in range(len(key))]))

    # change A(scipy csc matrix) into a numpy matrix
    adj_matrix_tmp = A.todense()
    # get the gender for each node(1/2,0 for missing)
    gender_y_tmp = metadata[:, 1]
    # get the corresponding gender for each node in a dictionary form
    gender_dict = create_dict(range(len(gender_y_tmp)), gender_y_tmp)

    (graph, gender_y_tmp) = parse_mat.create_graph(adj_matrix_tmp, gender_dict, 'gender', 0, None,
                                                   'yes')

    adj_matrix = nx.adjacency_matrix(graph).todense().tolist()

    # change A(scipy csc matrix) into a numpy matrix
    # adj_matrix = A.todense().tolist()
    # get the gender for each node (1/2, 0 for missing)
    # gender_y_tmp = metadata[:, 1]
    # gender_unknown = []
    gender_y = []
    for i, y in enumerate(gender_y_tmp):
        gender_y.append((i, y))

    return adj_matrix, gender_y


def write_files(adj_matrix, gender_y, random_seed=1, percent_labeled=0.01,
                data_name='Amherst41', learn=False):
    """

    :param adj_matrix:
    :param gender_y:
    :param random_seed: (str) random seed used to generate the data
    :param percent_labeled:
    :param data_name: (str) name of the school
    :param learn: (bool) specifies if the data will be used for learning or evaluation

    :return: nothing
    """

    # set up parameters
    params = {}
    params['timestamp'] = str(datetime.datetime.now())
    params['data'] = data_name
    params['% labeled'] = percent_labeled
    params['random seed'] = random_seed

    # set the working directory and import helper functions
    # get the current working directory and then redirect into the functions under code
    cwd = os.getcwd()

    # import the data from the data folder
    data_cwd = '{0}/{1}'.format(os.path.abspath(cwd), 'data')

    # set up a folder to hold our indicies (to use for the baseline model)
    indicies_cwd = '{0}/{1}'.format(os.path.abspath(cwd), 'index')
    Path(indicies_cwd).mkdir(parents=True, exist_ok=True)  # if directory doesn't exist, create it

    # new directory for learn/eval
    data_cwd = '{0}/{1}'.format(data_cwd, "learn" if learn else "eval")
    Path(data_cwd).mkdir(parents=True, exist_ok=True)  # if directory doesn't exist, create it

    # new directory for school
    data_cwd = '{0}/{1}'.format(data_cwd, data_name.split('.')[0])
    Path(data_cwd).mkdir(parents=True, exist_ok=True)  # if directory doesn't exist, create it

    # new directory for % labeled
    data_cwd = '{0}/{1:02d}pct'.format(data_cwd, int(percent_labeled * 100))
    Path(data_cwd).mkdir(parents=True, exist_ok=True)  # if directory doesn't exist, create it

    # new directory for random split
    data_cwd = '{0}/{1:04d}rand'.format(data_cwd,
                                        int(
                                            random_seed))  # TO-DO fix random seed size to 4 digits, append rand instead of preprending it
    Path(data_cwd).mkdir(parents=True, exist_ok=True)  # if directory doesn't exist, create it

    # collect the edges
    edges = []
    preferences_file = data_cwd + "/preferences_targets.txt"
    with open(preferences_file, 'w+') as f:
        for row_i, row in enumerate(adj_matrix):
            f.write('{0}\t1\n'.format(row_i))
            f.write('{0}\t2\n'.format(row_i))
            for col_i, column in enumerate(row):
                if column > 0:
                    edges.append((row_i, col_i))

    # write out the edge data
    edges_file = data_cwd + '/edges_obs'
    with open(edges_file + '.txt', 'w+') as f:
        for edge in edges:
            f.write('{0[0]}\t{0[1]}\n'.format(edge))

    gender_truth_file = data_cwd + '/gender_truth.txt'
    gender_obs_file = data_cwd + '/gender_obs.txt'
    gender_targets_file = data_cwd + '/gender_targets.txt'
    gender_train_indicies = indicies_cwd + '/gender_train_indicies_{}rand_{}pct.txt'.format(
        random_seed, percent_labeled)
    gender_test_indicies = indicies_cwd + '/gender_test_indicies_{}rand_{}pct.txt'.format(
        random_seed, percent_labeled)

    gender2 = gender_y.copy()
    random.shuffle(gender2)
    split = int(len(gender2) * percent_labeled)
    with open(gender_obs_file, 'w+') as f_obs, open(gender_targets_file, 'w+') as f_target, open(
            gender_truth_file, 'w+') as f_truth, open(gender_test_indicies, 'w+') as f_test_index, \
            open(gender_train_indicies, 'w+') as f_train_index:

        for gender_i, gender in (gender2[:split]):
            if gender > 0:
                f_train_index.write('{}\n'.format(gender_i))
                f_obs.write('{0}\t{1}\t{2}\n'.format(gender_i, 1, float(gender == 1)))
                f_obs.write('{0}\t{1}\t{2}\n'.format(gender_i, 2, float(gender == 2)))
        for gender_i, gender in (gender2[split:]):
            if gender > 0:
                f_test_index.write('{}\n'.format(gender_i))
                f_target.write('{0}\t1\n'.format(gender_i))
                f_target.write('{0}\t2\n'.format(gender_i))
                f_truth.write('{0}\t{1}\t{2}\n'.format(gender_i, 1, float(gender == 1)))
                f_truth.write('{0}\t{1}\t{2}\n'.format(gender_i, 2, float(gender == 2)))

        # for gender_i in gender_unknown:
        #     f_target.write('{0}\t1\n'.format(gender_i))
        #     f_target.write('{0}\t2\n'.format(gender_i))

    data_log = data_cwd + '/data_log.json'
    with open(data_log, 'w+') as f:
        json.dump(params, f)


if __name__ == "__main__":
    cli_parse = argparse.ArgumentParser()
    cli_parse.add_argument("--seed", help="Sets a random seed", default=1)
    cli_parse.add_argument("--data", help="Specifies the name of the data file to use",
                           default='Amherst41.mat')
    cli_parse.add_argument("--learn", dest='learn', action='store_true',
                           help='Specifies if this data will be used for learning or not')
    cli_parse.set_defaults(learn=False)
    args = cli_parse.parse_args()

    assert args.seed, "No random seed was provided"
    assert args.data, "No target data was provided"

    generate_data(args.seed, args.data, args.learn)
    # generate_data([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
