path1 = './res/1292_results.csv'
filename1 = path1
X1 = []
with open(filename1, 'r') as f:
    lines = f.readlines()
    for line in lines:
        value = [float(s) for s in line.split(',')]
        X1.append(value[0])

max_acc = 80
max40 = max_acc * 0.7
max60 = max_acc * 0.8
max80 = max_acc * 0.9

res4 = []
res6 = []
res8 = []
for i in range(len(X1)):
    if X1[i] > max40:
        res4.append(i)
    # if X1[i] > max60:
    #     res6.append(i)
    # if X1[i] > max80:
    #     res8.append(i)
print([res4[0]])