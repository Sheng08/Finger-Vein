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
#16的倍數
BS = 64

def compute_eer(fpr,tpr,thresholds):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]

def compute_eer2(far,frr,thresholds):
    far = np.array(far)
    frr = np.array(frr)
    abs_diffs = np.abs(far - frr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((far[min_index], frr[min_index]))
    return eer, thresholds[min_index]
def FAR_FRR(TN,FP, FN, TP):
    FAR = FP/(FP+TN)
    TPR = TP/(TP+FN)
    FRR = 1-TPR
    # if np.isnan(FAR):
    #     FAR =0
    # if np.isnan(FRR):
    #     FRR =0
    # if np.isnan(TPR):
    #     TPR =1
    return FAR, FRR ,TPR
	
def perf_measure2(model, x_test, y_test):

    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)

    truelabel = y_test.argmax(axis=-1)   # 將one-hot轉成對應label
    tr = label_binarize(truelabel, classes=[ i for i in range(123)])
    predict = model.predict(x_test)

    
    Min, Max = np.min(predict),  np.max(predict)
    print(-Min, Max)
    step_len = (Min + Max) / 1476
    # thred = np.arange(0, 1, 0.001)
    # # thred = np.append(predict, thred)
    # uniques = np.unique(thred)
    # y = np.argsort(-uniques)
    # thred = uniques[y]
    thred = np.arange(0, Max+(step_len*2), step_len)
    # thred = [i for i in range(Min, Max, 0.01)]
    print(thred)
    far = []
    frr = []
    tpr = []

    for threshold in thred:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(123):
            y_pred = (predict[:, i] >= threshold).astype('int32')
            # confusion_matrix(y_test, y_pred)
            # print("y_pred", y_pred)
            # print("y_test", tr[:, i])
            cm = confusion_matrix(y_true=tr[:, i], y_pred=y_pred, labels=[0,1])
            tn, fp, fn, tp = cm.ravel()
            TN = TN+tn
            FP = FP+fp
            FN = FN+fn
            TP = TP+tp
            # FP = (cm.sum(axis=0) - np.diag(cm))
            # FN = (cm.sum(axis=1) - np.diag(cm))
            # TP = np.diag(cm)
            # TN = cm.sum() - (FP + FN + TP)
            # TP = cm[0][0]+TP
            # FP = cm[0][1]+FP
            # FN = cm[1][0]+FN
            # TN = cm[1][1]+TN
            # print(FP,FN, TP, TN)

            # FP_sum = FP.sum()
            # FN_sum = FN.sum()
            # TP_sum = TP.sum()
            # TN_sum = TN.sum()
        print(threshold ,FAR_FRR(TN, FP, FN, TP))
        FAR, FRR, TPR = FAR_FRR(TN, FP, FN, TP)
        far.append(FAR)
        frr.append(FRR)
        tpr.append(TPR)

    roc_auc = metrics.auc(far, tpr)
    # fpr = np.array(far)
    # fnr = np.array(frr)
    # threshold = np.array(thred)
    # eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    # EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    EER , eer_threshold = compute_eer2(far, frr, thred)
    print("EER", EER)
    print("eer_threshold", eer_threshold)
    # print(thred)
    # print(far)
    # print(frr)
    # print(tpr)

    plt.figure()
    plt.plot(far, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc, marker = 'o',lw=2)
    plt.plot([-0.005, 1.005], [-0.005, 1.005], 'k--')
    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic', fontsize=18)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('./plot/ROC curve_class.png')#儲存圖片
    plt.show()
    
    plt.figure()
    plt.plot(frr, far, color = 'green', marker = 'o',label = 'ROC',lw=2)
    plt.legend()
    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])
    plt.xlabel('FRR')
    plt.ylabel('FAR')
    plt.grid(True)
    plt.title('ROC', fontsize=18)
    plt.savefig('./plot/ROC_class.png')#儲存圖片
    plt.show()
    
    plt.figure()
    plt.plot(thred, far,label = 'FAR', marker = '')
    plt.plot(thred, frr,label = 'FRR',marker = '')
    plt.plot(EER, eer_threshold,'ro', label = 'EER')
    plt.legend()
    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])
    plt.xlabel('thresh')
    plt.ylabel('FAR/FRR')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0,1),loc=1,borderaxespad=0.1)
    plt.suptitle('Find EER', fontsize=18)
    plt.title("EER = "+str(EER))
    plt.savefig('./plot/EER_class.png')#儲存圖片
    plt.show()



   
        # print(cm)
        # predict = model.predict(x_test)
        # predictions=np.argmax(predict,axis=1)
        # # predictions = model.predict_classes(x_test)
        # print(predict)
        # print(predictions)



    # threshold = 0.2
    # y_pred = (model.predict(X_test)[:, 1] > threshold).astype('int32')
    # # confusion_matrix(y_test, y_pred)
    # print(y_pred)
    # predict = model.predict(x_test)
    # predictions=np.argmax(predict,axis=1)
    # # predictions = model.predict_classes(x_test)
    # print(predict)
    # print(predictions)

    # truelabel = y_test.argmax(axis=-1)   # 將one-hot轉成對應label
    # predictions_score = model.predict(x_test)
    # np.set_printoptions(suppress=True)

    #predictions_score = np.round_(predictions_score, decimals = 3)
    #print("true:", truelabel)
    #print("pred:",predictions)
    #print("pr_s:",predictions_score)
	

    '''
    cm = confusion_matrix(y_true=truelabel, y_pred=predictions)
    print(cm)
    FP = (cm.sum(axis=0) - np.diag(cm))
    FN = (cm.sum(axis=1) - np.diag(cm))
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    print(FP,FN, TP, TN)

    FP_sum = FP.sum()
    FN_sum = FN.sum()
    TP_sum = TP.sum()
    TN_sum = TN.sum()

    FAR = FP_sum/FP_sum+TN_sum
    TPR = FP_sum/TP_sum+FN_sum
    FRR = 1-TPR
    '''


	
    # tr = label_binarize(truelabel, classes=[ i for i in range(123)])
    # #print(tr)
    # pr = label_binarize(predictions, classes=[ i for i in range(123)])
    # #print(pr)
    # n_classes = 123
    # fpr = dict()
    # fnr = dict()
    # tpr = dict()
    # thr = dict() 
    # roc_auc =  dict()
    # class_in_num = 12 #一類別內有多少測試張數
    # for i in range(n_classes):
    #     far = []
    #     frr = []
    #     abs_ = []
    #     #thred = []
    #     EER_X, EER_Y = 0, 0
    #     flag = 0
    #     ##p = predictions_score[:, i].copy()
    #     Min, Max = min(predictions_score[:, i]),  max(predictions_score[:, i])
    #     step_len = (- Min + Max) / 100
    #     thred = np.arange(-Min, Max, step_len)
    #     thred = np.append(predictions_score[:, i], thred)
    #     uniques = np.unique(thred)
    #     y = np.argsort(-uniques)
    #     thred = uniques[y]
    #     #print(thred)
    #     '''
    #     uniques = np.unique(predictions_score[:, i])
    #     thred = np.append(uniques, [max(uniques)])
    #     ##print(uniques)
    #     y = np.argsort(-thred)
    #     thred = thred[y]
    #     #print(thred)
	# 	'''
    #     #Min, Max = min(predictions_score[:, i]),  max(predictions_score[:, i])
    #     #step_len = (- Min + Max) / 10
    #     #thred = np.arange(-Min, Max, step_len)
    #     #Min, Max = min(predictions_score[:, i]), 1
    #     #for k in range(51):
    #     #    thred.append(Min + k * (Max - Min) / 50)
        
    #     fpr[i], tpr[i], thr[i] = metrics.roc_curve(tr[:, i], predictions_score[:, i], drop_intermediate=False)
    #     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
		
    #     #eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    #     #eer_thresh = interp1d(fpr, thresholds)(eer)
    #     #print("eer",eer)
    #     #print("eer_thresh", eer_thresh)
        # EER , eer_threshold = compute_eer(fpr[i], tpr[i], thr[i])
        # print("EER", EER)
        # print("eer_threshold", eer_threshold)
        # '''
        # fnr[i] = 1 - tpr[i]
        # eer_threshold = thr[i][np.nanargmin(np.absolute((fnr[i] - fpr[i])))]
        # EER = fnr[i][np.nanargmin(np.absolute((fnr[i] - fpr[i])))]
		# '''
        
        # plt.figure()
        # plt.plot(fpr[i], tpr[i], label='ROC curve (AUC = %0.2f)' % roc_auc[i], marker = '',lw=2)
        # plt.plot([-0.005, 1.005], [-0.005, 1.005], 'k--')
        # plt.xlim([-0.005, 1.005])
        # plt.ylim([-0.005, 1.005])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic class '+str(i+1), fontsize=18)
        # plt.legend(loc="lower right")
        # plt.grid(True)
        # plt.savefig('./plot/ROC curve_class '+str(i+1)+'.png')#儲存圖片
        # #plt.show()
        
        # plt.figure()
        # plt.plot(1-tpr[i], fpr[i], color = 'green', marker = '',label = 'ROC',lw=2)
        # plt.legend()
        # plt.xlim([-0.005, 1.005])
        # plt.ylim([-0.005, 1.005])
        # plt.xlabel('FRR')
        # plt.ylabel('FAR')
        # plt.grid(True)
        # plt.title('ROC class:'+str(i+1), fontsize=18)
        # plt.savefig('./plot/ROC_class '+str(i+1)+'.png')#儲存圖片
        # #plt.show()
		
        # plt.figure()
        # plt.plot(fpr[i], thr[i],label = 'FAR')
        # plt.plot(1-tpr[i], thr[i],label = 'FRR')
        # plt.plot(EER, eer_threshold,'ro', label = 'EER')
        # plt.legend()
        # plt.xlim([-0.005, 1.005])
        # plt.ylim([-0.005, 1.005])
        # plt.xlabel('thresh')
        # plt.ylabel('FAR/FRR')
        # plt.grid(True)
        # plt.legend(bbox_to_anchor=(1.0,1),loc=1,borderaxespad=0.1)
        # plt.suptitle('Find EER class:'+str(i+1), fontsize=18)
        # plt.title("EER = "+str(EER))
        # plt.savefig('./plot/EER_class '+str(i+1)+'.png')#儲存圖片
        # #plt.show()

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

# model = ResNet50(
# weights=None,
#     input_shape=(300, 100, 1),classes=123
# )


# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
# #                          height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
# #                          horizontal_flip=True, fill_mode="nearest")
# # history = model.fit_generator(
# #      aug.flow(X_train, Y_train, batch_size=BS),
# #      validation_data=(X_val, Y_val),
# #      steps_per_epoch=len(X_train) // BS,
# #      epochs=EPOCHS, verbose=1)

# history = model.fit(X_train, Y_train, validation_data=(
#    X_val, Y_val),  epochs=EPOCHS, batch_size=BS, verbose=1)



# X_test, Y_test = DataSet("./File1/TestF1_RawPohe_data/test_data", 12)  # X為影像 Y為標籤
# score, acc = model.evaluate(X_test, Y_test, batch_size=12)



# model.save('Resnet1.h5')


# # print(history.history.keys())
# plt.figure()
# plt.plot(history.history['acc'])  # 訓練準確度
# plt.plot(history.history['val_acc'])  # 驗證準確度
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.savefig('./plot/Model accuracy.png')#儲存圖片
# #plt.show()

# plt.figure()
# plt.plot(history.history['loss'])  # 訓練損失值
# plt.plot(history.history['val_loss'])  # 驗證損失值
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.savefig('./plot/Model loss.png')#儲存圖片
# #plt.show()


# print("Test score:", score)  # 測試
# print("Test accuracy:", acc)
# #"""

# 直接進行測試時 取消註解第280、282~287行觀察測試準確度
X_test, Y_test = DataSet("./File1/TestF1_RawPohe_data/test_data", 12)  # X為影像 Y為標籤
model = tf.keras.models.load_model('./Resnet1.h5')
# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# score, acc = model.evaluate(X_test, Y_test, batch_size=BS)
# print("Test score:", score)  # 測試
# print("Test accuracy:", acc)
perf_measure2(model, X_test, Y_test)
