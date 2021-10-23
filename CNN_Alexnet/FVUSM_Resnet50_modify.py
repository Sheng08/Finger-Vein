#!/usr/bin/env python
# coding: utf-8

import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


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
EPOCHS = 2000
#16的倍數
BS = 64

def compute_eer(fpr,tpr,thresholds):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]
	
def perf_measure2(model, x_test, y_test):
    predict = model.predict(x_test)
    predictions=np.argmax(predict,axis=1)
    # predictions = model.predict_classes(x_test)
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
    n_classes = 123
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
        plt.savefig('./resnetResult/ROC curve_class '+str(i+1)+'.png')#儲存圖片
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
        plt.savefig('./resnetResult/ROC_class '+str(i+1)+'.png')#儲存圖片
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
        plt.savefig('./resnetResult/EER_class '+str(i+1)+'.png')#儲存圖片
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


X, Y = DataSet("./File1/TestF1_RawPohe_data/train_data", 12)  # X為影像 Y為標籤


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





# 以訓練好的 ResNet50 為基礎來建立模型，
# 捨棄 ResNet50 頂層的 fully connected layers
net = ResNet50(include_top=False, weights=None, input_tensor=None,
                input_shape=(300, 100, 1), classes=123)
x = net.output
x = Flatten()(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

# 增加 Dense layer，以 softmax 產生個類別的機率值
output_layer = Dense(123, activation='softmax', name='softmax')(x)
net_final = Model(inputs=net.input, outputs=output_layer)


# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
# net_final.compile(optimizer=Adam(lr=1e-5),
#                   loss='categorical_crossentropy', metrics=['accuracy'])
net_final.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 輸出整個網路結構
net_final.summary()




aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

history = net_final.fit_generator(
     aug.flow(X_train, Y_train, batch_size=BS),
     validation_data=aug.flow(X_val, Y_val),
     steps_per_epoch=len(X_train) // BS,
     epochs=EPOCHS, verbose=1)

# history = model.fit(X_train, Y_train, validation_data=(
#    X_val, Y_val),  epochs=EPOCHS, batch_size=BS, verbose=1)



X_test, Y_test = DataSet("./File1/TestF1_RawPohe_data/test_data", 12)  # X為影像 Y為標籤
score, acc = net_final.evaluate(X_test, Y_test, batch_size=12)

net_final.save('Resnet1.h5')




# model = ResNet50(
# weights=None,
#     input_shape=(300, 100, 1),classes=123
# )


# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
#                          height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
#                          horizontal_flip=True, fill_mode="nearest")
# history = model.fit_generator(
#      aug.flow(X_train, Y_train, batch_size=BS),
#      validation_data=(X_val, Y_val),
#      steps_per_epoch=len(X_train) // BS,
#      epochs=EPOCHS, verbose=1)

# # history = model.fit(X_train, Y_train, validation_data=(
# #    X_val, Y_val),  epochs=EPOCHS, batch_size=BS, verbose=1)


# X_test, Y_test = DataSet("./File1/TestF1_RawPohe_data/test_data", 12)  # X為影像 Y為標籤
# score, acc = model.evaluate(X_test, Y_test, batch_size=12)


# model.save('Resnet1.h5')


# print(history.history.keys())
plt.figure()
plt.plot(history.history['acc'])  # 訓練準確度
plt.plot(history.history['val_acc'])  # 驗證準確度
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('./resnetResult/Model accuracy.png')#儲存圖片
#plt.show()

plt.figure()
plt.plot(history.history['loss'])  # 訓練損失值
plt.plot(history.history['val_loss'])  # 驗證損失值
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('./resnetResult/Model loss.png')#儲存圖片
#plt.show()


print("Test score:", score)  # 測試
print("Test accuracy:", acc)
#"""

# 直接進行測試時 取消註解第280、282~287行觀察測試準確度
# X_test, Y_test = DataSet("./File1/TestF1_RawPohe_data/test_data", 12)  # X為影像 Y為標籤
model = tf.keras.models.load_model('./Resnet1.h5')
# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# score, acc = model.evaluate(X_test, Y_test, batch_size=BS)
# print("Test score:", score)  # 測試
# print("Test accuracy:", acc)
perf_measure2(model, X_test, Y_test)
