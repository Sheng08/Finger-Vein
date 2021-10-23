#!/usr/bin/env python
# coding: utf-8

import os
import sys
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import itertools
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import metrics



seed = 3
np.random.seed(seed)
EPOCHS = 100
BS = 64

def compute_eer(fpr,tpr,thresholds):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]
	
def perf_measure2(model, x_test, y_test):
    predictions = model.predict_classes(x_test)
    truelabel = y_test.argmax(axis=-1)   # 將one-hot轉成對應label
    predictions_score = model.predict(x_test)
    np.set_printoptions(suppress=True)
    #predictions_score = np.round_(predictions_score, decimals = 3)
    #print("true:", truelabel)
    #print("pred:",predictions)
    #print("pr_s:",predictions_score)
	
    cm = confusion_matrix(y_true=truelabel, y_pred=predictions)
    print(cm)
    FP = (cm.sum(axis=0) - np.diag(cm))
    FN = (cm.sum(axis=1) - np.diag(cm))
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    print(FP,FN, TP, TN)
	
    tr = label_binarize(truelabel, classes=[ i for i in range(123)])
    #print(tr)
    pr = label_binarize(predictions, classes=[ i for i in range(123)])
    #print(pr)
    n_classes = 10
    fpr = dict()
    fnr = dict()
    tpr = dict()
    thr = dict() 
    roc_auc =  dict()
    class_in_num = 12 #一類別內有多少測試張數
    for i in range(n_classes):
        far = []
        frr = []
        abs_ = []
        #thred = []
        EER_X, EER_Y = 0, 0
        flag = 0
        ##p = predictions_score[:, i].copy()
        Min, Max = min(predictions_score[:, i]),  max(predictions_score[:, i])
        step_len = (- Min + Max) / 100
        thred = np.arange(-Min, Max, step_len)
        thred = np.append(predictions_score[:, i], thred)
        uniques = np.unique(thred)
        y = np.argsort(-uniques)
        thred = uniques[y]
        #print(thred)
        '''
        uniques = np.unique(predictions_score[:, i])
        thred = np.append(uniques, [max(uniques)])
        ##print(uniques)
        y = np.argsort(-thred)
        thred = thred[y]
        #print(thred)
		'''
        #Min, Max = min(predictions_score[:, i]),  max(predictions_score[:, i])
        #step_len = (- Min + Max) / 10
        #thred = np.arange(-Min, Max, step_len)
        #Min, Max = min(predictions_score[:, i]), 1
        #for k in range(51):
        #    thred.append(Min + k * (Max - Min) / 50)
        
        fpr[i], tpr[i], thr[i] = metrics.roc_curve(tr[:, i], predictions_score[:, i], drop_intermediate=False)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
		
        #eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        #eer_thresh = interp1d(fpr, thresholds)(eer)
        #print("eer",eer)
        #print("eer_thresh", eer_thresh)
        EER , eer_threshold = compute_eer(fpr[i], tpr[i], thr[i])
        print("EER", EER)
        print("eer_threshold", eer_threshold)
        '''
        fnr[i] = 1 - tpr[i]
        eer_threshold = thr[i][np.nanargmin(np.absolute((fnr[i] - fpr[i])))]
        EER = fnr[i][np.nanargmin(np.absolute((fnr[i] - fpr[i])))]
		'''
        
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (AUC = %0.2f)' % roc_auc[i], marker = '',lw=2)
        plt.plot([-0.005, 1.005], [-0.005, 1.005], 'k--')
        plt.xlim([-0.005, 1.005])
        plt.ylim([-0.005, 1.005])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic class '+str(i+1), fontsize=18)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('./plot/ROC curve_class '+str(i+1)+'.png')#儲存圖片
        #plt.show()
        
        plt.figure()
        plt.plot(1-tpr[i], fpr[i], color = 'green', marker = '',label = 'ROC',lw=2)
        plt.legend()
        plt.xlim([-0.005, 1.005])
        plt.ylim([-0.005, 1.005])
        plt.xlabel('FRR')
        plt.ylabel('FAR')
        plt.grid(True)
        plt.title('ROC class:'+str(i+1), fontsize=18)
        plt.savefig('./plot/ROC_class '+str(i+1)+'.png')#儲存圖片
        #plt.show()
		
        plt.figure()
        plt.plot(fpr[i], thr[i],label = 'FAR')
        plt.plot(1-tpr[i], thr[i],label = 'FRR')
        plt.plot(EER, eer_threshold,'ro', label = 'EER')
        plt.legend()
        plt.xlim([-0.005, 1.005])
        plt.ylim([-0.005, 1.005])
        plt.xlabel('thresh')
        plt.ylabel('FAR/FRR')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0,1),loc=1,borderaxespad=0.1)
        plt.suptitle('Find EER class:'+str(i+1), fontsize=18)
        plt.title("EER = "+str(EER))
        plt.savefig('./plot/EER_class '+str(i+1)+'.png')#儲存圖片
        #plt.show()

def DataSet(path, num):

    classes = len(os.listdir(path))
    print(len(os.listdir(path))*num)

    X = np.empty(
        (len(os.listdir(path))*num, 300, 100, 1))
    Y = np.empty(
        (len(os.listdir(path))*num, classes))

    count = 0
    labels = [0]*classes
    # print(len(labels))
    for folder in os.listdir(path):
        imgs = os.path.join(path, folder)
        # print(imgs)
        for i in os.listdir(imgs):
            # print(i)
            img_path = os.path.join(imgs, i)
            # print(img_path)
            img = image.load_img(img_path, target_size=(
                300, 100), color_mode="grayscale")
            # print(img)
            img = image.img_to_array(img) / 255.0

            X[count] = img
            # print(X_train)
            labels_copy = labels.copy()

            labels_copy[count//num] = 1
            # print(labels_copy)

            Y[count] = np.array(labels_copy)
            # print(count)
            # print(Y_train[count])
            count += 1
    print("Y", Y.shape)
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    Y = Y[index]
    return X, Y


X, Y = DataSet("./F1_clahe_dwt/train_data", 12)  # X為影像 Y為標籤


# 將原始資料91分(訓練、測試)，一部分作測試評估模型用
#(X_train, X_test, Y_train, Y_test) = train_test_split(X,
#                                                      Y, test_size=0.1, random_state=42)
#print("Y_test", Y_test.shape)
# 再將訓練資料82分(訓練、驗證)
(X_train, X_val, Y_train, Y_val) = train_test_split(X,
                                                    Y, test_size=0.2, random_state=42)

print('X_train shape : ', X_train.shape)
print('Y_train shape : ', Y_train.shape)
print('X_val shape : ', X_val.shape)
print('Y_val shape : ', Y_val.shape)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")


# # model Alexnet
model = Sequential()
model.add(Conv2D(96, (3, 3), strides=(4, 4), input_shape=(300, 100, 1),
                 padding='valid', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same',
                 activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                 activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                 activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                 activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
#model.add(Dense(1000, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # g參數1放class數 即輸出層
adam = optimizers.Adam(lr=0.0001)
#sgd = optimizers.SGD(lr=0.0009, decay=0.001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=adam, metrics=['accuracy'])
model.summary()


# # train
#history = model.fit(X_train, Y_train, validation_data=(
#    X_val, Y_val),  epochs=EPOCHS, batch_size=BS, verbose=1)
history = model.fit_generator(
     aug.flow(X_train, Y_train, batch_size=BS),
     validation_data=(X_val, Y_val),
     steps_per_epoch=len(X_train) // BS,
     epochs=EPOCHS, verbose=1)


# # evaluate
X_test, Y_test = DataSet("./F1_clahe_dwt/test_data", 12)  # X為影像 Y為標籤
score, acc = model.evaluate(X_test, Y_test, batch_size=32)


# # save
model.save('F1_clahe_dwt.h5')


# print(history.history.keys())
plt.figure()
plt.plot(history.history['acc'])  # 訓練準確度
plt.plot(history.history['val_acc'])  # 驗證準確度
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('./plot/Model accuracy.png')#儲存圖片
#plt.show()

plt.figure()
plt.plot(history.history['loss'])  # 訓練損失值
plt.plot(history.history['val_loss'])  # 驗證損失值
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('./plot/Model loss.png')#儲存圖片
#plt.show()


print("Test score:", score)  # 測試
print("Test accuracy:", acc)

#X_test, Y_test = DataSet("./F1/F1_clahe_dwt/test_data", 12)  # X為影像 Y為標籤
model = tensorflow.keras.models.load_model('F1_clahe_dwt.h5')
#score, acc = model.evaluate(X_test, Y_test, batch_size=32)

perf_measure2(model, X_test, Y_test)
