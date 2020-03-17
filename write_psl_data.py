import os
import random

import parsing as parse

# fix random
random.seed(1)

# set the working directory and import helper functions
# get the current working directory and then redirect into the functions under code
cwd = os.getcwd()
print(cwd)

# import the data from the data folder
data_cwd = os.path.abspath(cwd) + '\data'
print('data', data_cwd)
fb100_file = data_cwd + '\Amherst41'
print('help', fb100_file)
A, metadata = parse.parse_fb100_mat_file(fb100_file)

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

# collect the edges
edges = []
for row_i, row in enumerate(adj_matrix):
    for col_i, column in enumerate(row):
        if column > 0:
            edges.append((row_i, col_i))

# write out the edge data
edges_file = data_cwd + '\edges'
with open(edges_file + '.txt', 'w+') as f:
    for edge in edges:
        f.write('{0[0]}\t{0[1]}\n'.format(edge))

gender_truth_file = data_cwd + '\gender_truth'
gender_obs_file = data_cwd + '\gender_obs'

# write out truth values for gender data
percent_initially_unlabelled = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
with open(gender_truth_file + '.txt', 'w+') as f_truth:
    for gender_i, gender in gender_y:
        f_truth.write('{0}\t{1}\n'.format(gender_i, 1, int(gender == 1)))
        f_truth.write('{0}\t{1}\n'.format(gender_i, 0, int(gender == 0)))

# create observations and targets
for percent in percent_initially_unlabelled:
    with open(data_cwd + '\gender_{}_obs.txt'.format(percent), 'w+') as f_obs, open(
            data_cwd + '\gender_{}_targets.txt'.format(percent), 'w+') as f_target:
        gender2 = gender_y.copy()
        random.shuffle(gender2)
        split = int(len(gender2) * (1 - percent))
        for gender_i, gender in gender2[:split]:
            if gender > 0:
                f_obs.write('{0}\t{1}\t{2}\n'.format(gender_i, 1, int(gender == 1)))
                f_obs.write('{0}\t{1}\t{2}\n'.format(gender_i, 0, int(gender == 0)))
        for gender_i, gender in gender2[split:]:
            if gender > 0:
                f_target.write('{0}\n'.format(gender_i))
        # some genders are just unknown by the base data
        # we set these to be targets as well
        for gender_i in gender_unknown:
            f_target.write('{0}\n'.format(gender_i))

# with open(data_cwd + '\gender_targets.txt', 'w+') as f:
#    for i in range(len(adj_matrix)):
#        f.write('{}\n'.format(i))

print(type(adj_matrix))
print(gender_y_tmp.shape)
