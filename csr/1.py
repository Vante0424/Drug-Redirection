import numpy as np
from scipy.sparse import csr_matrix
import pickle

array_train = []
array_test = []
array_all = []

# train
f_train = open('../1-data-preprocessing/train/node_label0.txt')
for line in f_train.readlines():
    line = line.split()

    tmp = [0, 0, 0, 0]
    node = int(line[0])
    label = int(line[1])
    tmp[label] = 1

    array_train.append(tmp)
print(array_train)
f_train.close()


# test
f_test = open('../1-data-preprocessing/test/node_label0.txt')
for line in f_test.readlines():
    line = line.split()

    tmp = [0, 0, 0, 0]
    node = int(line[0])
    label = int(line[1])
    tmp[label] = 1

    array_test.append(tmp)
print(array_test)
f_test.close()


# all
f_all = open('../1-data-preprocessing/node_label0.txt')
for line in f_all.readlines():
    line = line.split()

    tmp = [0, 0, 0, 0]
    node = int(line[0])
    label = int(line[1])
    tmp[label] = 1

    array_all.append(tmp)
print(array_all)
f_all.close()


# pickle
f0 = open('ind0.drkg.y', 'wb')
csr_matrix_train = csr_matrix(array_train)
pickle.dump(csr_matrix_train, f0)

f1 = open('ind0.drkg.ty', 'wb')
csr_matrix_test = csr_matrix(array_test)
pickle.dump(csr_matrix_test, f1)

f2 = open('ind0.drkg.ally', 'wb')
csr_matrix_all = csr_matrix(array_all)
pickle.dump(csr_matrix_all, f2)
