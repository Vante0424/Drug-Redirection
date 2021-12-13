import random

# f1 = open('../data/drkg.tsv', 'r')
# f2 = open('drkg.txt', 'w')
#
# drkg = []
# for line in f1.readlines():
#     token = line.split()
#
#     if token[1].startswith('DRUGBANK'):
#         drkg.append(line)
#         f2.write(line)
#
# f1.close()
# f2.close()


f3 = open('drkg.txt', 'r', encoding='utf-8')
f4 = open('drkg0.txt', 'w', encoding='utf-8')

data = []
for line in f3.readlines():
    data.append(line)

select = [random.choice(data) for i in range(3000)]
for i in select:
    f4.write(str(i))
f3.close()
f4.close()