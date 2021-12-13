import numpy as np
import matplotlib.pyplot as plt


f1 = open('entity_embedding0.txt', 'r')
node_list = []
edge_dic = dict()
emb_x = []
emb_y = []

line_id = 0
node_num = 0
for line in f1.readlines():
    token = line.strip().split()
    print(len(token))

    node_id = int(line_id)
    node_list.append(node_id)
    emb_x.append(float(token[0]))
    emb_y.append(float(token[1]))
    pos = [token[0], token[1]]
    edge_dic[node_id] = pos

    line_id += 1
    node_num += 1

print(max(node_list))
print(min(node_list))
print(node_num)
if max(node_list) == node_num-1 and min(node_list) == 0:
    print(True)


emb_x = np.array(emb_x)
emb_y = np.array(emb_y)

plt.scatter(emb_x, emb_y)
# plt.show()
plt.savefig('TransE0.png')

f_out = open('out2-2_transe_embedding_drkg0.txt', 'w')
for node_id in range(node_num):
    pos1 = edge_dic[node_id][0]
    pos2 = edge_dic[node_id][1]
    out = pos1 + ',' + pos2 + '\n'
    f_out.write(out)
f_out.close()
