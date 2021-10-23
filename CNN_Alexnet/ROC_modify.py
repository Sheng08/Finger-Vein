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
    return FAR, FRR ,TPR
	
def perf_measure2(model, x_test, y_test):

    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)

    truelabel = y_test.argmax(axis=-1)   # 將one-hot轉成對應label
    tr = label_binarize(truelabel, classes=[ i for i in range(123)])
    predict = model.predict(x_test)

    
    Min, Max = np.min(predict),  np.max(predict)
    # print(-Min, Max)
    step_len = (Min + Max) / 1476
    thred = np.arange(0, Max+(step_len*2), step_len)
    # print(thred)
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
        # print(threshold ,FAR_FRR(TN, FP, FN, TP))
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