## Import statements

from imutils import paths
import os,cv2
import numpy as np
from scipy import interp
from itertools import cycle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import random
import itertools

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import regularizers
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras import regularizers
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model

from feature_aggregation import BagOfWords


from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import fetch_olivetti_faces
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


# ===================== variables =========================
#Random seeding
seed=7
np.random.seed(seed)

#Input size for VGG16
img_rows=224
img_cols=224
num_channel=3

#Training data directories
path_flower='../../train_data_patches/extended_flower_140'
path_leaf='../../train_data_patches/extended_leaf_140'


img_data_list=[]
cvscores=[]


def image_to_feature_vector(image, size=(img_rows, img_cols)):
		return cv2.resize(image, size).flatten()


# ===================== preparing data =========================
# Read images from the file system and flattening them
listing_flower=os.listdir(path_flower)
for file in listing_flower:
    input_img=cv2.imread(path_flower+'/'+file,1)
    input_img_flatten=image_to_feature_vector(input_img,(img_rows,img_cols))
    img_data_list.append(input_img_flatten)

listing_leaf=os.listdir(path_leaf)
for file in listing_leaf:
	input_img=cv2.imread(path_leaf+'/'+file,1)
	input_img_flatten=image_to_feature_vector(input_img,(img_rows,img_cols))
	img_data_list.append(input_img_flatten)

#Converting the list of images into a numpy array
img_data = np.array(img_data_list)

#Sanity check for image size
for i in img_data:
   if len(i) != 150528:
     print len(i)," ",

#Scale the images
img_data = img_data.astype('float32')
img_data_scaled = preprocessing.scale(img_data)
img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
img_data=img_data_scaled

num_of_samples = img_data.shape[0]
print num_of_samples

#Assign labels to the training data based on the directories they came from
labels = np.ones((num_of_samples,),dtype='int64')
labels[0:511]=1
labels[511:]=0

#VGG parameters
batch_size=25
nb_classes=2
nb_epoch=17
nb_filters=32
nb_pool=2
nb_conv=3

cvscores = []
conf=0
mean_tpr = 0.0
mean_fpr = np.linspace(0,1,100)
colors = cycle(['cyan','indigo','seagreen','yellow','blue'])
lw = 2
i = 0

# ===================== split data =========================
#Splitting data and performing Stratified k fold cross validation
x,y = shuffle(img_data,labels, random_state=2)
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)


for (train,test),color in zip(cv.split(x,y),colors):

	print len(train)
	print len(test)
	print color

	#Initialize descriptors
	des_list_train=[]
	des_list_test=[]

	#Get the already trained VGG16 model on ImageNet
	vgg16 = VGG16(weights='imagenet')
	fc2 = vgg16.get_layer('fc2').output

	model = Model(input=vgg16.input, output=fc2)
	model.summary()

	for layer in model.layers:
	    layer.trainable = False


	sgd = SGD(lr=1e-4, momentum=0.9)
	model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

	for j in x[train]:
	    des=j
	    des=des.reshape(1,img_rows,img_cols,3)
	    layer_name = 'fc1'
	    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
	    intermediate_output = intermediate_layer_model.predict(des)
	    des_list_train.append((j,intermediate_output))


	descriptors_train = des_list_train[0][1]
	for image_path, descriptor in des_list_train[1:]:
	    descriptors_train = np.vstack((descriptors_train, descriptor))

	for p in x[test]:
	    des=p
	    des=des.reshape(1,img_rows,img_cols,3)
	    layer_name = 'fc1'
	    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
	    intermediate_output = intermediate_layer_model.predict(des)
	    des_list_test.append((p,intermediate_output))


	descriptors_test = des_list_test[0][1]
	for image_path, descriptor in des_list_test[1:]:
	    descriptors_test = np.vstack((descriptors_test, descriptor))

	#Initialize an SVM classifier
	svm = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape=None, probability=True, degree=2, gamma='auto', kernel='rbf',verbose=False)

	clf=svm.fit(descriptors_train,y[train])

	##Accuracy
	score=clf.score(descriptors_test,y[test])
	cvscores.append(score)
	print score

	##Confusion matrix
	conf1=confusion_matrix(y[test],clf.predict(descriptors_test))
	conf=conf+conf1

	####ROC curve
	probas_ = clf.fit(descriptors_train, y[train]).predict_proba(descriptors_test)
	fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])

	fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
	mean_tpr += interp(mean_fpr, fpr, tpr)
	mean_tpr[0] = 0.0
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, lw=lw, color=color,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
	i += 1


mean_score=np.mean(cvscores)
std_score=np.std(cvscores)

#Printing the results
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
mean_tpr /= cv.get_n_splits(x, y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE FOR VGG')
plt.legend(loc="lower right")
plt.savefig('vgg_roc.png')
plt.show()
print("####################################")
