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

# read data
# data1 = pd.read_csv('data_ICS_6_TT_multi.csv')
# data2 = pd.read_csv('data_ICS_6_TT_multi.csv')
multiple = 0

data1 = pd.read_csv('./train_test/data_ICS_6_TT_multi_train_shuffle.csv')
data2 = pd.read_csv('./train_test/data_ICS_6_TT_multi_test_shuffle.csv')

# dummy encode labels, store separately
# 1D
# labels_full1 = data1['type']
# labels_full2 = data2['type']
# 2D
labels_full1 = pd.get_dummies(data1['type'], prefix='type')
labels_full2 = pd.get_dummies(data2['type'], prefix='type')

# drop labels from training dataset
data1 = data1.drop(columns='type')
data2 = data2.drop(columns='type')

# training data for the neural net
train_data_st = data1.values
train_data_nd = data2.values

t = StandardScaler()
train_data_st = t.fit_transform(train_data_st)
train_data_nd = t.transform(train_data_nd)

# labels for training
labels1 = labels_full1.values
labels2 = labels_full2.values

# Validation Technique
# x_train1, x_test1, y_train1, y_test1 = train_test_split(train_data_st, labels1, test_size=0.2)
# x_train2, x_test2, y_train2, y_test2 = train_test_split(train_data_nd, labels2, test_size=0.2)
# x_train1, x_validate1, y_train1, y_validate1 = train_test_split(x_train1, y_train1, test_size=0.125)

x_train1 = train_data_st
y_train1 = labels1
x_test2 = train_data_nd
y_test2 = labels2
x_train1, x_validate1, y_train1, y_validate1 = train_test_split(x_train1, y_train1, test_size=0.125)

# Build Model
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# modelName
modelName = 'STL-BOOST'

# train
clf = xgb.XGBClassifier()
clf.fit(x_train1, y_train1)
# test
y_predicted = clf.predict(x_test2)
print("y_test\n", y_test2)
print("y_predicted\n", y_predicted)
accuracy = clf.score(x_test2, y_test2)
print("Accuracy: {:.2f}%".format(accuracy * 100))
'''
print(y_test2.argmax(axis=1))
print(y_predicted.argmax(axis=1))
fpr, tpr, threshold = roc_curve(y_test2.argmax(axis=1), y_predicted.argmax(axis=1))
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of ' + modelName)
plt.show()
'''

# plot CM
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# model = tf.keras.models.load_model('/'+model_name+'.h5')
y_pred = clf.predict(x_test2)
y_pred_cm = np.argmax(y_pred, axis=1)
y_test_cm = np.argmax(y_test2, axis=1)
cm = confusion_matrix(y_test_cm, y_pred_cm)
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
labels = np.asarray(labels).reshape(4, 4)
label = ['benign', 'SYN', 'UDP', 'TCP']
# plt.figure(figsize=(4, 4))
sns.heatmap(cm, xticklabels=label, yticklabels=label, annot=labels, annot_kws={'size': 14}, fmt='', cmap="Blues")
plt.title('Confusion Matrix for ' + modelName + ' Model', fontsize='14')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('True Class', fontsize=14)
plt.xlabel('Predicted Class', fontsize=14)
plt.savefig('./ans/' + modelName + '_CM_{}.jpg'.format(multiple), dpi=600, bbox_inches='tight')
plt.show()

# plot ACC、Recall、F1, plot CR
print(classification_report(y_test_cm, y_pred_cm, target_names=['benign', 'SYN', 'UDP', 'TCP']))
with open('./ans/' + modelName + '_CR_{}.txt'.format(multiple), 'a') as f:
    f.write(classification_report(y_test_cm, y_pred_cm, target_names=['benign', 'SYN', 'UDP', 'TCP']))

# plot ROC
from itertools import cycle

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(labels.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test2[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['red', 'purple', 'blue', 'green', 'aqua', 'violet'])
for i, color in zip(range(labels.shape[1]), colors):
    plt.plot(fpr[i], tpr[i], color=color, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
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
plt.savefig('./ans/' + modelName + '_ROC_{}.jpg'.format(multiple), dpi=600, bbox_inches='tight')
plt.show()

with open('./savemodel/multi/STL-BOOST.pickle', 'wb') as f:
    pickle.dump(clf, f)
