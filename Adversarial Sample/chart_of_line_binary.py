import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# one model with all noise
MN = ['IoT-Node ANN', 'DR-1DCN', 'IoT BKCM', 'PCA+MI', 'ML-DT', 'RF-IDS', 'PSO-SVM', 'STL-BOOST']

for mn in range(len(MN)):
    modelName = MN[mn]
    Accuracy = []
    Precision = []
    Recall = []
    F1 = []
    for i in range(0, 11):
        M = i / 20
        if i == 0:
            M = 0
        # if i == 10:
        #     M = 1
        # M = i
        f = open('./result/result_Real/binary/' + modelName + '/' + modelName + '_CR_{}.txt'.format(M))
        data = f.readlines()
        s1 = ''
        for j in range(39, 43):
            s1 += data[5][j]
        s2 = ''
        for j in range(19, 23):
            s2 += data[6][j]
        s3 = ''
        for j in range(29, 33):
            s3 += data[6][j]
        s4 = ''
        for j in range(39, 43):
            s4 += data[6][j]
        Accuracy.append(float(s1))
        Precision.append(float(s2))
        Recall.append((float(s3)))
        F1.append((float(s4)))
        f.close()

    # x = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # x = [0, 1, 2, 3, 4, 5]
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # x = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    # Accuracy =  [0.99, 0.94, 0.90, 0.88, 0.87, 0.85, 0.84, 0.83, 0.83, 0.82, 0.81]
    # Precision = [0.98, 0.95, 0.93, 0.93, 0.92, 0.91, 0.91, 0.90, 0.90, 0.90, 0.90]
    # Recall =    [0.99, 0.88, 0.80, 0.76, 0.73, 0.71, 0.69, 0.66, 0.64, 0.63, 0.63]
    # F1 =        [0.99, 0.91, 0.84, 0.81, 0.77, 0.75, 0.72, 0.69, 0.67, 0.65, 0.65]

    # ax = plt.axes()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    # xx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # xx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # xx = [0, 1, 2, 3, 4, 5]
    # xx = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    xx = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    # yy = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    # yy = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    yy = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    plt.xticks(x, xx)
    # plt.yticks(yy)

    plt.plot(x, Accuracy)
    plt.plot(x, Precision)
    plt.plot(x, Recall)
    plt.plot(x, F1)

    plt.xlim(0, 1)
    # plt.ylim(0, 1)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Binary Classification Results of ' + modelName + ' Model', fontsize=14)
    plt.xlabel('Magnitude of Noise', fontsize=14)
    plt.ylabel('Result', fontsize=14)
    plt.legend(['Accuracy', 'Precision', 'Recall', 'F1'], loc='upper right')
    plt.grid('on')
    plt.savefig('./result/result_Pic/line_chart/' + modelName + '/' + 'linechart_binary_' + modelName + '.jpg', dpi=600,
                bbox_inches='tight')
    plt.show()
