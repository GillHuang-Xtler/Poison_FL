import matplotlib.pyplot as plt

def plt_txt():
    filename = 'iid-bart.txt'
    X,Y = [],[]
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split()]
            X.append(value[0])
    print(X)

    plt.plot(X)
    plt.show()

def plt_acc():
    path1 = './res/1171_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])
    print(X1)

    path2 = './res/1172_results.csv'
    filename2 = path2
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])
    print(X2)

    path3='./res/1191_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[0])
    print(X3)

    path4='./res/1175_results.csv'
    filename4 = path4
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[0])
    print(X4)
    # plt.plot(X4)
    #
    path5='./res/1176_results.csv'
    filename5 = path5
    X5 = []
    with open(filename5, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X5.append(value[0])
    print(X5)

    path6='./res/1173_results.csv'
    filename6 = path6
    X6 = []
    with open(filename6, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X6.append(value[0])
    print(X6)

    path7='./res/1177_results.csv'
    filename7 = path7
    X7 = []
    with open(filename7, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X7.append(value[0])
    print(X7)

    path8='./res/1178_results.csv'
    filename8 = path8
    X8 = []
    with open(filename8, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X8.append(value[0])
    print(X8)

    path9='./res/1179_results.csv'
    filename9 = path9
    X9 = []
    with open(filename9, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X9.append(value[0])
    print(X9)

    path10='./res/1170_results.csv'
    filename10 = path10
    X10 = []
    with open(filename10, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X10.append(value[0])
    print(X10)

    plt.plot(X1, color='yellow', label='IID')
    plt.plot(X2, color='black', label='Reduce-class')
    plt.plot(X3, color='brown', label='Reduce-class-plus')
    # plt.plot(X4, color='green', label='Reduce-class-only')
    # plt.plot(X5, color='orange', label='Reduce-class-musto')
    plt.plot(X6, color='purple', label='Reduce-class-mustp')
    # plt.plot(X7, color='green', label='Reduce-class-quitp')
    # plt.plot(X8, color='skyblue', label='Reduce-class-quito')
    # plt.plot(X9, color='purple', label='Reduce-class-introp')
    # plt.plot(X10, color='gray', label='Reduce-class-introo')


    plt.legend()
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('ACCURACY')

    plt.show()

def plt_class_recall_1():
    path1 = './res/1171_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[13])

    print(X1)

    path2 = './res/1172_results.csv'
    filename2 = path2
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[13])
    print(X2)

    path3 = './res/1191_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[13])
    print(X3)

    path4 = './res/1091_results.csv'
    filename4 = path4
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[13])
    print(X4)
    # plt.plot(X4)

    path5='./res/1113_results.csv'
    filename5 = path5
    X5 = []
    with open(filename5, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X5.append(value[13])
    print(X5)

    path6='./res/1122_results.csv'
    filename6 = path6
    X6 = []
    with open(filename6, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X6.append(value[13])
    print(X6)

    path7='./res/1177_results.csv'
    filename7 = path7
    X7 = []
    with open(filename7, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X7.append(value[13])
    print(X7)

    path9='./res/1179_results.csv'
    filename9 = path9
    X9 = []
    with open(filename9, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X9.append(value[13])
    print(X9)

    plt.plot(X1, color='yellow', label='IID', linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X2, color='black', label='Reduce-class',linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X3, color='brown', label='Reduce-class-plus',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X4, color='skyblue', label='Reduce-class-only',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X5, color='orange', label='Reduce-class-must',linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X6, color='purple', label='Reduce-class-mustp',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X7, color='green', label='Reduce-class-quitp',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X9, color='purple', label='Reduce-class-introp',linestyle=':', marker = 'o', markersize = 2)

    plt.legend()
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('SOURCE CLASS RECALL')
    plt.show()

def find_count(X1):
    max_acc = 75
    max40 = max_acc * 0.7
    max60 = max_acc * 0.8
    max80 = max_acc * 0.9

    res4 = []
    res6 = []
    res8 = []
    for i in range(len(X1)):
        if X1[i] > max40:
            res4.append(i)
        if X1[i] > max60:
            res6.append(i)
        if X1[i] > max80:
            res8.append(i)
    return [res4[0], res6[0], res8[0]]


def nun_maverick():

    path3 = './res/1191_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[13])

    path4 = './res/1091_results.csv'
    filename4 = path4
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[13])


    path5 = './res/1113_results.csv'
    filename5 = path5
    X5 = []
    with open(filename5, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X5.append(value[13])

    name_list = ['Random', 'FedFast', 'New']
    num_list1 = []
    num_list2 = []
    num_list3 = []
    num_list1.extend([X3[0], X4[0], X5[0]])
    num_list2.extend([X3[1], X4[1], X5[1]])
    num_list3.extend([X3[2], X4[2], X5[2]])

    x = list(range(len(num_list1)))
    total_width, n = 0.8, 2
    width = total_width / n

    plt.bar(x, num_list1, width=width, label='1 maverick', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list2, width=width, label='2 maverick', tick_label=name_list, fc='r')
    plt.bar(x, num_list3, width=width, label='3 maverick', tick_label=name_list, fc='r')

    plt.legend()
    plt.show()

    # path6='./res/1122_results.csv'
    # filename6 = path6
    # X6 = []
    # with open(filename6, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         value = [float(s) for s in line.split(',')]
    #         X6.append(value[13])
    # print(X6)
    #
    # path7='./res/1177_results.csv'
    # filename7 = path7
    # X7 = []
    # with open(filename7, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         value = [float(s) for s in line.split(',')]
    #         X7.append(value[13])
    # print(X7)
    #
    # path9='./res/1179_results.csv'
    # filename9 = path9
    # X9 = []
    # with open(filename9, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         value = [float(s) for s in line.split(',')]
    #         X9.append(value[13])
    # print(X9)

    plt.plot(X1, color='yellow', label='IID', linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X2, color='black', label='Reduce-class',linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X3, color='brown', label='Reduce-class-plus',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X4, color='skyblue', label='Reduce-class-only',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X5, color='orange', label='Reduce-class-must',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X6, color='purple', label='Reduce-class-mustp',linestyle=':', marker = 'o', markersize = 2)

def plt_utility():
    path1 = './shapley-cifar.txt'
    filename1 = path1
    X1, Y1 = [0], [1]
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])
            Y1.append(1-((abs(value[1]/value[6]-0.1)+abs(value[2]/value[6]-0.1)+abs(value[3]/value[6]-0.1)+abs(value[4]/value[6]-0.1)+abs(value[5]/value[6]-0.6))))
    X2 = ['REF', 'AVG']
    Y2 = [1, sum(Y1[1:])/(len(Y1)-1)]

    grid = plt.GridSpec(1, 4)
    plt.subplot(grid[0,0:3])
    plt.bar(X1[0],  Y1[0], color = 'r', width=4)
    plt.bar(X1[1:], Y1[1:], color = 'b', width=4)
    plt.plot(X1[1:], Y1[1:], color = 'g',marker = 'o', markersize = 4)
    plt.legend()
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('FAIRNESS UTILITY')
    plt.subplot(grid[0,3])
    plt.bar(X2[0], Y2[0], color = 'r', width=0.5)
    plt.bar(X2[1], Y2[1], color = 'b', width=0.5)
    plt.legend()
    # plt.xlabel('GLOBAL ROUNDS')
    plt.tight_layout()
    plt.show()
    # print(X1, Y1)

def sub_plot():
    path1 = './res/1013_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])

    path2 = './res/1314_results.csv'
    filename2 = path2
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])

    path3 = './res/1012_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[0])

    path4 = './res/1313_results.csv'
    filename4 = path4
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[0])

    path6='./res/1215_results.csv'
    filename6 = path6
    X6 = []
    with open(filename6, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X6.append(value[0])

    path7 = './res/3315_results.csv'
    filename7 = path7
    X7 = []
    with open(filename7, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X7.append(value[0])

    path8='./res/1253_results.csv'
    filename8 = path8
    X8 = []
    with open(filename8, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X8.append(value[0])

    path9='./res/3314_results.csv'
    filename9 = path9
    X9 = []
    with open(filename9, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X9.append(value[0])
    plt.figure(22)
    plt.subplot(221)
    plt.plot(X1, color='blue', label='sv', linewidth='1')
    plt.plot(X2, color='red', label='new', linewidth='1')
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('ACCURACY')
    plt.ylim([40,75])
    plt.legend()

    plt.subplot(222)
    plt.plot(X3, color='blue', label='sv', linewidth='1')
    plt.plot(X4, color='red', label='new', linewidth='1')
    plt.xlabel('GLOBAL ROUNDS')
    plt.legend()

    plt.subplot(223)
    plt.plot(X8, color='red', label='new', linewidth='1')
    plt.plot(X9, color='blue', label='sv', linewidth='1')
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('ACCURACY')
    plt.ylim([40,75])
    plt.legend()

    plt.subplot(224)
    plt.plot(X6, color='red', label='new', linewidth='1')
    plt.plot(X7, color='blue', label='sv', linewidth='1')
    plt.xlabel('GLOBAL ROUNDS')
    # plt.ylabel('ACCURACY')
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ =='__main__':
    # plt_txt()
    # plt_acc()
    # plt_class_recall_1()
    # plt_utility()
    # sub_plot()
    path1 = './res/2881_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])
    path2 = './res/2881_results.csv'
    filename2 = path2
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])

    path3 = './res/1171_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[0])


    # plt.plot(X1, color='brown', label='avg', linewidth = '1')
    plt.plot(X2, color='blue', label='het', linewidth = '1')
    plt.plot(X3, color='green', label='iid', linewidth = '1')

    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('ACCURACY')
    # plt.ylim([60,95])
    plt.legend()
    plt.show()