from collections import defaultdict
import pickle
import networkx as nx

f = open('../1-data-preprocessing/graph_edges0.txt')
graph = defaultdict(list)
node = -1
tmp = []
for line in f.readlines():
    line = line.split()
    if line[0] == 'node_id':
        continue
    elif len(line) == 0:
        continue
    else:
        if int(line[0]) != node:
            node = int(line[0])
            tmp = []
        tmp.append(int(line[1]))
    graph[node] = tmp
    # exit()
print(graph)

f0 = open('ind0.drkg.graph', 'wb')
pickle.dump(graph, f0)
f0.close()


with open('ind0.drkg.graph', 'rb') as f:
    a = pickle.load(f, encoding='latin1')
print(a)

# adj = nx.adjacency_matrix(nx.from_dict_of_lists(a))

# s = {52: ['25'], 25: ['72'], 19: ['117'], 44: ['19']}
# t = {52: [25], 25: [72], 19: [117], 44: [19]}
# r = nx.Graph(t)
# print(r.nodes())
# adj = nx.adjacency_matrix(r)
# print(adj)
