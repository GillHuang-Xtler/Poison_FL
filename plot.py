import matplotlib.pyplot as plt

def plt_txt():
    filename = 'test.txt'
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
    path1 = './test.txt'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])
    print(X1)

    path2 = './res/111_results.csv'
    filename2 = path2
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])
    print(X2)

    path3='./res/1082_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[0])
    print(X3)

    path4='./res/1091_results.csv'
    filename4 = path4
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[0])
    print(X4)
    # plt.plot(X4)

    path5='./res/1113_results.csv'
    filename5 = path5
    X5 = []
    with open(filename5, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X5.append(value[0])
    print(X5)

    path6='./res/1122_results.csv'
    filename6 = path6
    X6 = []
    with open(filename6, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X6.append(value[0])
    print(X5)

    plt.plot(X1, color='green', label='IID')
    plt.plot(X2, color='skyblue', label='Reduce-class')
    plt.plot(X3, color='red', label='Reduce-class-plus')
    plt.plot(X4, color='blue', label='Reduce-class-only')
    plt.plot(X5, color='orange', label='Reduce-class-must')
    plt.plot(X6, color='brown', label='Reduce-class-mustp')

    plt.legend()
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('ACCURACY')

    plt.show()

def plt_class_recall_1():
    path1 = './label-flipping-attack_d-fashion-mnist_n-cnn_p-0_e-full_m-Influence_500_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[13])

    print(X1)

    path2 = './res/111_results.csv'
    filename2 = path2
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[13])
    print(X2)

    path3 = './res/1082_results.csv'
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

    plt.plot(X1, color='green', label='IID', linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X2, color='skyblue', label='Reduce-class',linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X3, color='red', label='Reduce-class-plus',linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X4, color='blue', label='Reduce-class-only',linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X5, color='orange', label='Reduce-class-must',linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X5, color='brown', label='Reduce-class-mustp',linestyle=':', marker = 'o', markersize = 2)

    plt.legend()
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('SOURCE CLASS RECALL')
    plt.show()

if __name__ =='__main__':
    # plt_txt()
    plt_acc()
    # plt_class_recall_1()
    # path1 = './res/1101_results.csv'
    # filename1 = path1
    # X1 = []
    # with open(filename1, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         value = [float(s) for s in line.split(',')]
    #         X1.append(value[0])
    #
    # print(X1)
    # plt.plot(X1)
    # plt.show()