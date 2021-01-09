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
    path = './res/111_results.csv'
    filename = path
    X = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X.append(value[0])
    print(X)

    plt.plot(X)
    plt.show()

def plt_class_recall_1(path = './res/111_results.csv'):
    filename = path
    X = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X.append(value[16])
    print(X)

    plt.plot(X)
    plt.show()

if __name__ =='__main__':
    # plt_txt()
    # plt_acc(path='./res/1082_results.csv')
    plt_class_recall_1(path='./label-flipping-attack_d-fashion-mnist_n-cnn_p-0_e-full_m-Influence_500_results.csv')