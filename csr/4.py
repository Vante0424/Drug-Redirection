f1 = open('../1-data-preprocessing/entities0.dict', 'r')
f2 = open('ind0.drkg.test.index', 'w')
f3 = open('../1-data-preprocessing/test/entities0.dict', 'r')

all = {}
text = []
for line in f1.readlines():
    line = line.strip().split()
    all[line[1]] = line[0]
print(len(all))

for line in f3.readlines():
    line = line.strip().split()
    if line[1] in all:
        text.append(int(all[line[1]]))
print(len(text))

text.sort()
for i in text:
    f2.write(str(i) + '\n')
