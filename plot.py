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

if __name__ =='__main__':
    # plt_txt()
    # plt_acc()
    # plt_class_recall_1()

    path1 = './res/1244_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])

    path2 = './res/1243_results.csv'
    filename2 = path2
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])
    #
    # path2 = './res/cifarShapley.txt'
    # filename2 = path2
    # X2 , Y2= [], []
    # with open(filename2, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         value = [float(s) for s in line.split()]
    #         X2.append(value[0])
    #         Y2.append(value[1]/value[2]*200+10)
    #         print(X2, Y2)
    #
    #
    # plt.plot(X1, Y1, color='purple', label='Reduce-class-plus',linestyle=':', marker = 'o', markersize = 2 )
    # plt.plot(X2, Y2, color = 'purple', linestyle=':', marker = 'o', markersize = 2)
    print(X1, X2)
    plt.plot(X1, color = 'brown')
    plt.plot(X2, color = 'blue')

    # plt.ylim(40,100)
    plt.show()