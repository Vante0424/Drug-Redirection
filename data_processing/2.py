
f1 = open('node_id_mapping0.txt', 'r')
f2 = open('entities0.dict', 'w')

for line in f1.readlines():
    token = line.split()

    f2.write(token[1] + '\t' + token[0] + '\n')

f1.close()
f2.close()
