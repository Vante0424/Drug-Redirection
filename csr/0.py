import numpy as np
from scipy.sparse import csr_matrix
import pickle

# train
array_train = []
f_train = open('../1-data-preprocessing/train/train_embedding0.txt', 'r')
for line in f_train.readlines():
    line = line.split()
    array_train.append(line)

f_train.close()


# test
array_test = []
f_test = open('../1-data-preprocessing/test/test_embedding0.txt', 'r')
for line in f_test.readlines():
    line = line.split()
    array_test.append(line)

f_test.close()


# all
array_all = []
f_all = open('../2-embedding/entity_embedding0.txt', 'r')
for line in f_all.readlines():
    line = line.split()
    array_all.append(line)
f_all.close()


array_train = np.array(array_train)
array_test = np.array(array_test)
array_all = np.array(array_all)
array_train = np.array(array_train).astype(np.float64)
array_test = np.array(array_test).astype(np.float64)
array_all = np.array(array_all).astype(np.float64)
csr_matrix_train = csr_matrix(array_train)
csr_matrix_test = csr_matrix(array_test)
csr_matrix_all = csr_matrix(array_all)
# print(type(csr_matrix_train))


f0 = open('ind0.drkg.x', 'wb')
pickle.dump(array_train, f0)
f0.close()

f1 = open('ind0.drkg.tx', 'wb')
pickle.dump(array_test, f1)
f1.close()

f2 = open('ind0.drkg.allx', 'wb')
pickle.dump(array_all, f2)
f2.close()


