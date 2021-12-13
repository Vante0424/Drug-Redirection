import numpy as np


def read_edges():
    # i = 0
    edge_list = []
    node_set = set()
    neibor_dic = dict()
    f = open('../1-data-preprocessing/graph_edges0.txt', 'r')
    for line in f.readlines():
        # i += 1
        ele = line.strip().split('\t')
        if ele[0] == 'node_id':
            continue
        edge_list.append(ele)
        node_set.add(ele[0])
        node_set.add(ele[1])
        if ele[0] == ele[1]:
            continue
        if ele[0] not in neibor_dic:
            neibor_dic[ele[0]] = set()
        if ele[1] not in neibor_dic:
            neibor_dic[ele[1]] = set()
        neibor_dic[ele[0]].add(ele[1])
        neibor_dic[ele[1]].add(ele[0])

        # if i >= 5:
        #     break

    f.close()
    node_list = list(map(int, node_set))
    if max(node_list) == len(node_list)-1 and min(node_list) == 0:
        print('True')
    graph_neibor_dic = neibor_dic

    # print(node_set)
    # print(edge_list)
    # print(graph_neibor_dic)

    return node_set, edge_list, graph_neibor_dic


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
        # print(pos)

        # if count >= 5:
        #     break

    f.close()
    # print(pos_dic)
    print('read_pos OK')
    return pos_dic


def cal_dist_Eucd(pos1, pos2):
    v = np.array(pos1)
    u = np.array(pos2)
    z = v - u
    # print(pos1, pos2)
    # print(v)
    # print(u)
    # print(z)
    # exit()
    return np.sqrt(np.einsum('i,i->', z, z))


def find_space_neibor(node_id, pos_dic, r):
    space_neibor_set = set()
    pos_node_1 = pos_dic[node_id]
    for target_id in pos_dic.keys():
        dist = cal_dist_Eucd(pos_node_1, pos_dic[target_id])
        # print(dist)
        # exit()
        if dist < r and target_id != node_id:
            space_neibor_set.add(target_id)
    return space_neibor_set


def space_neibor(node_set, pos_dic, r):
    print('start space_neibor')
    space_neibor_dic = dict()
    average_degree = 0
    # i = 0
    for node_id in node_set:
        # i += 1
        space_neibor_set = find_space_neibor(node_id, pos_dic, r)
        # print(len(space_neibor_set))
        space_neibor_dic[node_id] = space_neibor_set
        average_degree += len(space_neibor_set)

        # if i >= 5:
            # break

    # print(space_neibor_dic)
    average_degree = average_degree*1.0/len(node_set)
    print('space_neibor OK')
    return space_neibor_dic, average_degree


def averge_degree_graph(node_set, neibor_dic):
    average_degree = 0
    for node_id in node_set:
        neibor_set = neibor_dic[node_id]
        average_degree += len(neibor_set)
    average_degree = average_degree*1.0/len(node_set)
    return average_degree


node_set, edge_list, graph_neibor_dic = read_edges()
file_name = '../2-embedding/out2-2_transe_embedding_drkg0.txt'
pos_dic = read_pos(file_name)

r = 0.0385
space_neibor_dic, average_degree_space = space_neibor(node_set, pos_dic, r)
print(average_degree_space)
# exit()
average_degree_graph = averge_degree_graph(node_set, graph_neibor_dic)
print(average_degree_graph)

# exit()

f_out = open('outf_nodes_space_relation_drkg_transe0.txt', 'w')
illus = 'node1,node2\tspace\trelation_type\n'
f_out.write(illus)

for node_id in node_set:
    source_pos = pos_dic[node_id]

    for tar_id in graph_neibor_dic[node_id]:
        target_pos = pos_dic[tar_id]
        if source_pos[0] > target_pos[0]:  # determine the relative position relationship of two nodes
            if source_pos[1] > target_pos[1]:
                relation_type = 0
            else:
                relation_type = 1
        else:
            if source_pos[1] > target_pos[1]:
                relation_type = 2
            else:
                relation_type = 3

        node_st = list(map(str, [node_id, tar_id]))
        space_type = 'graph'
        out = ','.join(node_st) + '\t' + space_type + '\t' + str(relation_type) + '\n'
        f_out.write(out)

    for tar_id in space_neibor_dic[node_id]:
        target_pos = pos_dic[tar_id]
        if source_pos[0] > target_pos[0]:  # determine the relative position relationship of two nodes
            if source_pos[1] > target_pos[1]:
                relation_type = 0
            else:
                relation_type = 1
        else:
            if source_pos[1] > target_pos[1]:
                relation_type = 2
            else:
                relation_type = 3

        node_st = list(map(str, [node_id, tar_id]))
        space_type = 'latent_space'
        out = ','.join(node_st) + '\t' + space_type + '\t' + str(relation_type) + '\n'
        f_out.write(out)
