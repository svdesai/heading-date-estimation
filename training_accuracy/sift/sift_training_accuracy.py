import numpy as np
import cv2
from sklearn.metrics import classification_report

from feature_aggregation import BagOfWords
import random
from itertools import cycle

import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import metrics

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from keras import regularizers
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
k_list=[5,50]

cvscores=[]
for p in k_list:

    ###Getting sift descriptors
    def sift(*args, **kwargs):
        try:
            return cv2.xfeatures2d.SIFT_create(*args, **kwargs)
        except:
            return cv2.SIFT()



    ###Getting feature descriptos for an image at every pixel with a distance of 15 in between
    def dsift(img, step=p):
        keypoints = [
            cv2.KeyPoint(x, y, step)
            for y in range(0, img.shape[0], step)
            for x in range(0, img.shape[1], step)
        ]
        features = sift().compute(img, keypoints)[1]
         
        norm=features.sum(axis=1).reshape(-1, 1)
        norm[norm == 0] = 0.0001
        features /= norm
        return features




 #   path2='/Users/manasakumar/Downloads/actual_resized_train'

    #path2a='/Users/manasakumar/Downloads/two_new/extended_flower_resized'
    #path2b='/Users/manasakumar/Downloads/two_new/extended_leaf_resized'
    
    path2a = '/home/coderfreak2/myml/new_ds/extended_flower_140'
    path2b = '/home/coderfreak2/myml/new_ds/extended_leaf_140'



    ##for later
    seed=7
    np.random.seed(seed)


    ###Preparing the images!
    img_rows=140
    img_cols=140
    img_data_list=[]
    num_channel=1



    def image_to_feature_vector(image, size=(img_rows, img_cols)):
                return cv2.resize(image, size).flatten()



    listing2a=os.listdir(path2a)
    for file in listing2a:
        input_img=cv2.imread(path2a+'/'+file,0)
        input_img_flatten=image_to_feature_vector(input_img,(img_rows,img_cols))
        img_data_list.append(input_img_flatten)


    listing2b=os.listdir(path2b)
    for file in listing2b:
        input_img=cv2.imread(path2b+'/'+file,0)
        input_img_flatten=image_to_feature_vector(input_img,(img_rows,img_cols))
        img_data_list.append(input_img_flatten)


    ###Preparing samples
    img_data = np.array(img_data_list)
    num_of_samples = img_data.shape[0]
    print num_of_samples




    ### Preparing labels
    labels = np.ones((num_of_samples,),dtype='int64')

    labels[0:512]=1
    labels[512:]=0


    arr1=[]
    cvscores=[]
    conf=0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue'])
    lw = 2
    i = 0


    x,y = shuffle(img_data,labels, random_state=2)



    kfold = StratifiedKFold(n_splits=5, random_state=7)



    for (train, test),color in zip(kfold.split(x,y),colors):
            print len(train)
            print len(test)
            print color






            features_train = [
                dsift((k.reshape(img_rows,img_cols,1)*255).astype(np.uint8))
                for k in x[train]
            ]






            bow_train = BagOfWords(800)


            bow_train.fit(features_train)
            #
            img_bow_train = bow_train.transform(features_train)



            svm = SVC(probability=True,kernel=chi2_kernel).fit(img_bow_train,y[train])

 #           clf=svm.fit(img_bow_train, y[train])



            features_test = [
                dsift((n.reshape(img_rows,img_cols,1)*255).astype(np.uint8))
                for n in x[test]
            ]

            img_bow_test = bow_train.transform(features_test)


            from scipy import interp

            from sklearn.metrics import roc_auc_score


            ####Accuracy
            score=svm.score(img_bow_test,y[test])
            cvscores.append(score)
            print score


            ####Confusion matrix
            conf1=confusion_matrix(y[test],svm.predict(img_bow_test))
            conf=conf+conf1

            ####ROC curve
            probas_ = svm.fit(img_bow_train, y[train]).predict_proba(img_bow_test)
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])

            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=lw, color=color,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
            i += 1




 #           arr2=arr2+svm.predict(img_bow_test)
 #           print arr1
 #           print arr2
            #print(metrics.precision_score(arr1, arr2))

            #print(classification_report(y[test], svm.predict(img_bow_test)))





    mean_score=np.mean(cvscores)
    std_score=np.std(cvscores)

    print("####################################")
    print("Accuracy:")
    print mean_score
    print ("+/-")
    print std_score

    print("####################################")
    print("Confusion Matrix:")
    print conf
    print("####################################")
    print("ROC AND AUC")
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',label='Luck')
    mean_tpr /= kfold.get_n_splits(x, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC CURVE FOR SIFT D = '+str(p)+' pixels')
    plt.legend(loc="lower right")
    plt.show()
    print("####################################")




##
