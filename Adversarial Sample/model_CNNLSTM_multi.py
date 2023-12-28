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

x_train_cnn1 = np.reshape(x_train1, (x_train1.shape[0], x_train1.shape[1], 1))
# x_test_cnn1 = np.reshape(x_test1, (x_test1.shape[0], x_test1.shape[1], 1))
x_test_cnn2 = np.reshape(x_test2, (x_test2.shape[0], x_test2.shape[1], 1))
x_validate_cnn1 = np.reshape(x_validate1, (x_validate1.shape[0], x_validate1.shape[1], 1))

# Build Model
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=5, strides=1, padding='same', input_shape=(train_data_st.shape[1], 1)))
model.add(Conv1D(filters=64, kernel_size=5, strides=1, padding='same'))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(16, return_sequences=True))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(labels1.shape[1], activation='softmax'))
modelName = 'IoT BKCM'
# keras.utils.plot_model(model, './'+modelName+'_Archi.jpg',show_shapes=True)
model.summary()

adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            lr=0.00001)
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=10,
                          verbose=1,
                          restore_best_weights=True)

checkpoint = ModelCheckpoint('./savemodel/multi/' + modelName + '.h5',
                            # ('./ans/' + modelName + '_{}.h5'.format(multiple),
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             # save_weights_only=True,
                             verbose=1)

epochs = 30
batch_size = 512
history = model.fit(x_train_cnn1, y_train1, batch_size=batch_size,
                    steps_per_epoch=x_train_cnn1.shape[0] // batch_size,
                    epochs=epochs,
                    validation_data=(x_validate_cnn1, y_validate1),
                    # validation_split=0.10,
                    callbacks=[learning_rate_reduction, checkpoint]
                    )


# plot model's validation loss and validation accuracy
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'], '--*',
                color=(1, 0, 0))
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'], '--^',
                color=(0.7, 0, 0.7))
    axs[0].set_title('Model ' + modelName + ' Accuracy', fontsize='14')
    axs[0].set_ylabel('Accuracy', fontsize='14')
    axs[0].set_xlabel('Epoch', fontsize='14')
    # axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1), len(model_history.history['accuracy']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[0].grid('on')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'], '--x',
                color=(0, 0.5, 0))
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'], '--D',
                color=(0, 0, 0.5))
    axs[1].set_title('Model ' + modelName + ' Loss', fontsize='14')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    axs[1].set_ylabel('Loss', fontsize='14')
    axs[1].set_xlabel('Epoch', fontsize='14')
    # axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    axs[1].grid('on')
    plt.savefig('./ans/' + modelName + '.jpg', dpi=600, bbox_inches='tight')
    plt.show()


'''
def plot_model_history(model_history):
    # summarize history for accuracy
    plt.plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'], '--*',
             color='red')
    plt.plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'], '--^',
             color='purple')
    plt.title('Model ' + modelName + ' Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'val'], loc='best')
    plt.grid('on')
    plt.savefig('./ans/' + modelName + ' Accuracy' + '_{}.jpg'.format(multiple), dpi=600, quality=100, optimize=True)
    plt.show()

    # summarize history for loss
    plt.plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'], '--x',
             color='blue')
    plt.plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'], '--D',
             color='green')
    plt.title('Model ' + modelName + ' Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'], loc='best')
    plt.grid('on')
    plt.savefig('./ans/' + modelName + ' Loss' + '_{}.jpg'.format(multiple), dpi=600, quality=100, optimize=True)
    plt.show()
'''


plot_model_history(history)
with open('./ans/History_' + modelName + '_{}'.format(multiple), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc

# model = tf.keras.models.load_model('/'+model_name+'.h5')
y_pred = model.predict(x_test_cnn2)
y_pred_cm = np.argmax(y_pred, axis=1)
y_test_cm = np.argmax(y_test2, axis=1)
cm = confusion_matrix(y_test_cm, y_pred_cm)
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
labels = np.asarray(labels).reshape(4, 4)
label = ['benign', 'SYN', 'UDP', 'TCP']
# plt.figure(figsize=(5, 5))
sns.heatmap(cm, xticklabels=label, yticklabels=label, annot=labels, annot_kws={'size': 14}, fmt='', cmap="Blues")
plt.title('Confusion Matrix for ' + modelName + ' Model', fontsize='14')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('True Class', fontsize=14)
plt.xlabel('Predicted Class', fontsize=14)
plt.savefig('./ans/' + modelName + '_CM_{}.jpg'.format(multiple), dpi=600, bbox_inches='tight')
plt.show()

print(classification_report(y_test_cm, y_pred_cm, target_names=['benign', 'SYN', 'UDP', 'TCP']))
loss, accuracy = model.evaluate(x_test_cnn2, y_test2, verbose=1)
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
with open('./ans/' + modelName + '_CR_{}.txt'.format(multiple), 'a') as f:
    f.write(classification_report(y_test_cm, y_pred_cm, target_names=['benign', 'SYN', 'UDP', 'TCP']))
    f.write("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

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
plt.savefig('./ans/' + modelName + '_ROC_{}.jpg'.format(multiple), dpi=600, bbox_inches='tight')
plt.show()
