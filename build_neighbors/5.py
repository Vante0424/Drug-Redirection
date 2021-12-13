import numpy as np
import heapq
import random


def read_edges():
    edge_list = []
    node_set = set()
    neibor_dic = dict()
    f = open('../1-data-preprocessing/graph_edges0.txt', 'r')
    for line in f.readlines():
        ele = line.strip().split('\t')
        if ele[0] == 'node_id':
            continue
        edge_list.append(ele)
        node_set.add(ele[0])
        node_set.add(ele[1])
        if ele[0] not in neibor_dic:
            neibor_dic[ele[0]] = set()
        if ele[1] not in neibor_dic:
            neibor_dic[ele[1]] = set()
        neibor_dic[ele[0]].add(ele[1])
        neibor_dic[ele[1]].add(ele[0])
    f.close()
    node_list = list(map(int, node_set))
    if max(node_list) == len(node_list)-1 and min(node_list) == 0:
        print('True')
    return node_set, edge_list, neibor_dic


def read_label():
    label_dic = dict()
    f = open('../1-data-preprocessing/node_label0.txt', 'r')
    for line in f.readlines():
        ele = line.strip().split('\t')
        if ele[0] == 'node_id':
            continue
        node_id = ele[0]
        label = ele[1]
        label_dic[node_id] = label
    f.close()
    return label_dic


def cal_homophili_graph(node_set, neibor_dic, label_dic):
    value = 0
    average_degree = 0
    for node_id in node_set:
        node_label = label_dic[node_id]
        neibor_set = neibor_dic[node_id]
        if node_id in neibor_set:
            neibor_set.remove(node_id)
        neibor_label_list = []
        for neibor_id in neibor_set:
            neibor_label = label_dic[neibor_id]
            neibor_label_list.append(neibor_label)
        num_of_same_label = neibor_label_list.count(node_label)
        same_label_ratio = num_of_same_label*1.0 / len(neibor_label_list)
        value += same_label_ratio
        average_degree += len(neibor_set)
    average_same_label_ratio = value/len(node_set)
    average_degree = average_degree*1.0/len(node_set)
    return average_same_label_ratio, average_degree


def read_pos(file_name):
    pos_dic = dict()
    f = open(file_name, 'r')
    count = 0
    for line in f.readlines():
        ele = line.strip().split(',')
        node_id = str(count)
        count += 1
        pos = list(map(float, ele))
        pos_dic[node_id] = pos
    f.close()
    return pos_dic


def cal_dist_Eucd(pos1, pos2):
    vec1 = np.array(pos1)
    vec2 = np.array(pos2)
    distance = np.sqrt(np.sum(np.square(vec1-vec2)))
    return distance


def cal_dist_space(pos_dic):
    dist_matrix = []
    for node1 in range(len(pos_dic)):
        node1_id = str(node1)
        node1_pos = pos_dic[node1_id]
        dist_list = []
        for node2 in range(len(pos_dic)):
            node2_id = str(node2)
            node2_pos = pos_dic[node2_id]
            dist = cal_dist_Eucd(node1_pos, node2_pos)
            dist_list.append(dist)
        dist_matrix.append(dist_list)
    return dist_matrix


def top_k_nearest(k, dist_list):
    nsmallestList = heapq.nsmallest(k+1, dist_list)
    nearest_k_dist_list = [dist for dist in nsmallestList if dist != 0]
    top_k_id_list = [dist_list.index(dist) for dist in nsmallestList if dist != 0]
    if len(top_k_id_list) != len(set(top_k_id_list)):
        print('top_k_error')
    top_k_id_list = list(map(str, top_k_id_list))
    return top_k_id_list, nearest_k_dist_list


def cal_homophili_space_topK(node_set, dist_matrix, label_dic, K):
    value = 0
    for node_id in node_set:
        node_label = label_dic[node_id]
        top_k_id_list, nearest_k_dist_list = top_k_nearest(K, dist_matrix[int(node_id)])
        neibor_label_list = []
        for neibor_id in top_k_id_list:
            neibor_label = label_dic[neibor_id]
            neibor_label_list.append(neibor_label)
        num_of_same_label = neibor_label_list.count(node_label)
        same_label_ratio = num_of_same_label * 1.0 / len(neibor_label_list)
        value += same_label_ratio
    average_same_label_ratio = value / len(node_set)
    return average_same_label_ratio


def cal_homophili_space_area(node_set, dist_matrix, label_dic, r):
    value = 0
    count = 0
    average_neighbor = 0
    for node_id in node_set:
        node_label = label_dic[node_id]
        dist_list = dist_matrix[int(node_id)]
        near_id_list = [dist_list.index(dist) for dist in dist_list if (dist < r and dist != 0)]
        if len(near_id_list) != len(set(near_id_list)):
            print('are_error')
        near_id_list = list(map(str, near_id_list))
        neibor_label_list = []
        if near_id_list == []:
            continue
        count += 1
        for neibor_id in near_id_list:
            neibor_label = label_dic[neibor_id]
            neibor_label_list.append(neibor_label)
        num_of_same_label = neibor_label_list.count(node_label)
        same_label_ratio = num_of_same_label * 1.0 / len(neibor_label_list)
        value += same_label_ratio
        average_neighbor += len(near_id_list)
    average_same_label_ratio = value / len(node_set)
    average_neighbor = average_neighbor/count
    return average_same_label_ratio, average_neighbor


def find_r(average_degree, node_set, dist_matrix):
    r_list = []
    degree_int = int(average_degree)
    degree_float = average_degree - degree_int
    for node_id in node_set:
        tmp = random.random()
        if tmp < degree_float:
            K = degree_int
        else:
            K = degree_int + 1
        top_k_id_list, nearest_k_dist_list = top_k_nearest(K, dist_matrix[int(node_id)])
        r_list.append(max(nearest_k_dist_list))
    median_r = np.median(r_list)
    return median_r


node_set, edge_list, neibor_dic = read_edges()
label_dic = read_label()
average_same_label_ratio, average_degree = cal_homophili_graph(node_set, neibor_dic, label_dic)
# print('average_degree', average_degree)
# print('homophili_graph', average_same_label_ratio)


file_name = '../2-embedding/out2-2_transe_embedding_drkg0.txt'
pos_dic = read_pos(file_name)
dist_matrix = cal_dist_space(pos_dic)
for k in [int(average_degree), int(average_degree)+1]:
    average_same_label_ratio = cal_homophili_space_topK(node_set, dist_matrix, label_dic, k)
    # print(k, average_same_label_ratio)


r = 0.0385
average_same_label_ratio, average_neighbor = cal_homophili_space_area(node_set, dist_matrix, label_dic, r)
print(r, average_same_label_ratio, average_neighbor)
