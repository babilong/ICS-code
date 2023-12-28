import random
import math
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.model_selection import *
from sklearn.preprocessing import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Conv1D, LSTM, Flatten, Add, Activation, \
    MaxPooling1D
from tensorflow.keras.models import Model
import tensorflow as tf
import pickle
import scipy.io as scio

# modelName
modelName = 'ML-DT'

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from itertools import cycle

# test
with open('./savemodel/binary/ML-DT.pickle', 'rb') as f:
    clf = pickle.load(f)

for multiple in range(0, 11):

    M = multiple / 20
    if multiple == 0:
        M = 0
    # if multiple == 10:
    #     M = 1

    # M = multiple
    data_test = pd.read_csv('dataset/binary-test/data_ICS_6_TT_binary_noise_{}.csv'.format(M))

    labels_full_test = pd.get_dummies(data_test['type'], prefix='type')
    data_test = data_test.drop(columns='type')
    train_data_test = data_test.values
    labels_test = labels_full_test.values

    # t = StandardScaler()
    # train_data_test = t.fit_transform(train_data_test)

    # x_train2, x_test2, y_train2, y_test2 = train_test_split(train_data_test, labels_test, test_size=0.2)
    x_test2 = train_data_test
    y_test2 = labels_test

    y_predicted = clf.predict(x_test2)
    accuracy = clf.score(x_test2, y_test2)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    # print(y_test2.argmax(axis=1))
    # print(y_predicted.argmax(axis=1))

    # plot CM
    y_pred = clf.predict(x_test2)
    y_pred_cm = np.argmax(y_pred, axis=1)
    y_test_cm = np.argmax(y_test2, axis=1)
    cm = confusion_matrix(y_test_cm, y_pred_cm)
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    label = ['benign', 'vicious']
    # plt.figure(figsize=(4, 4))
    sns.heatmap(cm, xticklabels=label, yticklabels=label, annot=labels, annot_kws={'size': 14}, fmt='', cmap="Blues")
    plt.title('Confusion Matrix for ' + modelName + ' Model', fontsize='14')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.savefig('./ans/' + modelName + '_CM_{}.jpg'.format(M), dpi=600, bbox_inches='tight')
    plt.show()

    # plot ACC、Recall、F1, plot CR
    print(classification_report(y_test_cm, y_pred_cm, target_names=['benign', 'vicious']))
    with open('./ans/' + modelName + '_CR_{}.txt'.format(M), 'a') as f:
        f.write(classification_report(y_test_cm, y_pred_cm, target_names=['benign', 'vicious']))

    # plot ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(labels.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test2[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['red', 'purple', 'blue', 'green', 'aqua', 'violet'])
    # for i, color in zip(range(labels.shape[1]), colors):
    for i, color in zip(range(0, 1), colors):
        # plt.plot(fpr[i], tpr[i], color=color, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot(fpr[i], tpr[i], color=color, label='ROC curve (area = {0:0.2f})'.format(roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.ylabel('Recall')
    # plt.xlabel('Fall-out (1-Specificity)')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('True Positive Rate', fontsize='14')
    plt.xlabel('False Positive Rate', fontsize='14')
    plt.title('ROC for ' + modelName + ' Model', fontsize='14')
    plt.legend(loc="lower right")
    plt.savefig('./ans/' + modelName + '_ROC_{}.jpg'.format(M), dpi=600, bbox_inches='tight')
    plt.show()
