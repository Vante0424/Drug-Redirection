import networkx as nx
from sklearn.model_selection import train_test_split


f1 = open('drkg0.txt', 'r')

node_list = []
edge_list = []
label_set = set()
label_dict = dict()


for line in f1.readlines():
    token = line.strip().split('\t')

    node1 = token[0]
    node2 = token[2]

    node_list.append(node1)
    node_list.append(node2)

    edge_list.append([node1, node2])
    edge_list.append([node2, node1])

    node_token1 = node1.split('::')
    node_token2 = node2.split('::')
    label_set.add(node_token1[0])
    label_set.add(node_token2[0])

    label_dict[node1] = node_token1[0]
    label_dict[node2] = node_token2[0]

f1.close()


node_list = list(set(node_list))
# print(len(node_list))
# print(len(edge_list))
# print(label_set)
# print(label_dict)


G = nx.Graph()
G.add_nodes_from(node_list)
G.add_edges_from(edge_list)


# print(nx.is_connected(G))
# print(nx.number_connected_components(G))


largest_components = max(nx.connected_components(G), key=len)
# print(largest_components)
# print(len(node_list))


node_in_comp_list = largest_components
edge_in_comp_list = [edge for edge in edge_list if edge[0] in node_in_comp_list]
# print(edge_in_comp_list)


f0 = open('edge_in_comp_list0.txt', 'w')
for pair in edge_in_comp_list:
    f0.write(pair[0] + ' ' + pair[1] + '\n')
f0.close()

# print(len(edge_list))
# node_in_comp_list = list(node_in_comp_list)
# print(node_in_comp_list[3])

# node_id mapping
node_mapping_dic = dict()
new_id = 0

for original_id in node_in_comp_list:
    node_mapping_dic[original_id] = new_id
    new_id += 1
# print(new_id)


f_out_1 = open('node_id_mapping0.txt', 'w')
f_out_1.write('original_node_id\tnew_node_id\n')
for original_id in node_mapping_dic:
    new_id = str(node_mapping_dic[original_id])
    out = original_id + '\t' + new_id + '\n'
    f_out_1.write(out)
f_out_1.close()


f_out_2 = open('graph_edges.txt0', 'w')
f_out_2.write('node_id\tnode_id\n')
for edge in edge_in_comp_list:
    node1 = edge[0]
    node2 = edge[1]
    node1_new = str(node_mapping_dic[node1])
    node2_new = str(node_mapping_dic[node2])
    out = node1_new + '\t' + node2_new + '\n'
    f_out_2.write(out)
f_out_2.close()


# label_id mapping
label_mapping_dic = dict()
label_id = 0

for ele in label_set:
    if ele not in label_mapping_dic:
        label_mapping_dic[ele] = label_id
        label_id += 1

f_out_3 = open('label_mapping0.txt', 'w')
f_out_3.write('label\tlabel_id\n')
for key in label_mapping_dic:
    original_id = str(key)
    label_id = str(label_mapping_dic[key])
    out = original_id + '\t' + label_id + '\n'
    f_out_3.write(out)
f_out_3.close()


f_out_4 = open('node_label0.txt', 'w')
f_out_4.write('node_id\tlabel\n')
for node_ori_id in node_in_comp_list:
    new_node_id = str(node_mapping_dic[node_ori_id])
    label = label_dict[node_ori_id]
    label_id = str(label_mapping_dic[label])
    out = new_node_id + '\t' + label_id + '\n'
    f_out_4.write(out)
f_out_4.close()


# Split datasets
drkg = []
f1 = open('drkg0.txt', 'r')
for line in f1.readlines():
    token = line.strip().split('\t')

    if token[0] in largest_components and token[2] in largest_components:
        drkg.append(line)

f1.close()
print(len(drkg))


def train_test_val_split(df, ratio_train, ratio_test, ratio_val):
    train, middle = train_test_split(df, test_size=1-ratio_train)
    ratio = ratio_val / (1-ratio_train)
    test, validation = train_test_split(middle, test_size=ratio)
    return train, test, validation


train, test, val = train_test_val_split(drkg, 0.8, 0.1, 0.1)

with open('train/train0.txt', 'w') as f:
    for i in train:
        f.write(i)

with open('test/test0.txt', 'w') as f:
    for i in test:
        f.write(i)

with open('valid/valid0.txt', 'w') as f:
    for i in val:
        f.write(i)
