# ===================== imports =========================

from imutils import paths
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import pandas as pd
import datetime
from scipy import interp
from itertools import cycle
from timeit import default_timer as timer
import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD

from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_olivetti_faces
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from feature_aggregation import BagOfWords
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix

# ===================== variables =========================

seed=7
np.random.seed(seed)

#Set the paths to the training patches
path_train_flower='../../../extended_kfold/fifth/train/flower'
path_train_leaf='../../../extended_kfold/fifth/train/leaf'

path_test_flower='../../../extended_kfold/fifth/validation/flower'
path_test_leaf='../../../extended_kfold/fifth/validation/leaf'

img_rows=224
img_cols=224
img_data_list_train=[]

img_data_list_test=[]

num_channel=3
num_classes = 2
cvscores=[]

# ===================== preparing data =========================

def image_to_feature_vector(image, size=(img_rows, img_cols)):
        return cv2.resize(image, size).flatten()

listing_train_flower=os.listdir(path_train_flower)
for file in listing_train_flower:
    input_img=cv2.imread(path_train_flower+'/'+file)
    input_img_flatten=image_to_feature_vector(input_img,(img_rows,img_cols))
    img_data_list_train.append(input_img_flatten)

listing_train_leaf=os.listdir(path_train_leaf)
for file in listing_train_leaf:
    input_img=cv2.imread(path_train_leaf+'/'+file)
    input_img_flatten=image_to_feature_vector(input_img,(img_rows,img_cols))
    img_data_list_train.append(input_img_flatten)

img_data_train = np.array(img_data_list_train)
img_data_train = img_data_train.astype('float32')
img_data_scaled_train = preprocessing.scale(img_data_train)
img_data_scaled_train=img_data_scaled_train.reshape(img_data_train.shape[0],img_rows,img_cols,3)
img_data_train=img_data_scaled_train

num_of_samples_train = img_data_train.shape[0]

labels_train = np.ones((num_of_samples_train,),dtype='int64')
labels_train[0:408]=1
labels_train[408:]=0

listing_test_flower=os.listdir(path_test_flower)
for file in listing_test_flower:
    input_img=cv2.imread(path_test_flower+'/'+file)
    input_img_flatten=image_to_feature_vector(input_img,(img_rows,img_cols))
    img_data_list_test.append(input_img_flatten)

listing_test_leaf=os.listdir(path_test_leaf)
for file in listing_test_leaf:
    input_img=cv2.imread(path_test_leaf+'/'+file)
    input_img_flatten=image_to_feature_vector(input_img,(img_rows,img_cols))
    img_data_list_test.append(input_img_flatten)

img_data_test = np.array(img_data_list_test)
img_data_test = img_data_test.astype('float32')
img_data_scaled_test = preprocessing.scale(img_data_test)
img_data_scaled_test=img_data_scaled_test.reshape(img_data_test.shape[0],img_rows,img_cols,3)
img_data_test=img_data_scaled_test

num_of_samples_test = img_data_test.shape[0]

labels_test = np.ones((num_of_samples_test,),dtype='int64')
labels_test[0:102]=1
labels_test[102:]=0

batch_size=25
nb_classes=2
nb_epoch=17
nb_filters=32
nb_pool=2
nb_conv=3

des_list_train=[]
des_list_test=[]

# ===================== Compiling the VGG model =========================

vgg16 = VGG16(weights='imagenet')
fc2 = vgg16.get_layer('fc2').output
model = Model(input=vgg16.input, output=fc2)
model.summary()

for layer in model.layers:
    layer.trainable = False

sgd = SGD(lr=1e-4, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# ======================== preparing the data for VGG and getting their features ==========================
start = timer()
for j in img_data_train:
    des=j
    des=des.reshape(1,img_rows,img_cols,3)
    layer_name = 'fc1'
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(des)
    des_list_train.append((j,intermediate_output))

descriptors_train = des_list_train[0][1]
for image_path, descriptor in des_list_train[1:]:
    descriptors_train = np.vstack((descriptors_train, descriptor))

for p in img_data_test:
    des=p
    des=des.reshape(1,img_rows,img_cols,3)
    layer_name = 'fc1'
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(des)
    des_list_test.append((p,intermediate_output))

descriptors_test = des_list_test[0][1]
for image_path, descriptor in des_list_test[1:]:
    descriptors_test = np.vstack((descriptors_test, descriptor))

# ======================== training the SVM ==========================

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape=None, probability=True, degree=3, gamma='auto', kernel='poly',verbose=False)
clf=clf.fit(descriptors_train,labels_train)
end = timer()

print('Training time: '+str(end-start)+' secs')
# ======================== calculating accuracy with test data ==========================

# ======================== Sliding window code ==========================
df = pd.DataFrame({'Name': [],'Value':[]})
#Specify the path to the test data : (kinmaze dataset, in this case)
path_for_test_images = sys.argv[1]
path_for_output_images='boxed_images' ##Save the output images containing bounding boxes

global k
k=0

listing3=os.listdir(path_for_test_images)
listing3 = sorted(listing3)

for file in listing3:
    global k
    global coord
    coord=[]
    img=cv2.imread(path_for_test_images+'/'+file)
    resized=img
    wind_row, wind_col = 140,140
    img_rows, img_cols = 140,140

    def image_to_feature_vector(image, size=(img_rows, img_cols)):
        return cv2.resize(image, size).flatten()

    def sliding_window(image, stepSize, windowSize):
          for y in range(0, image.shape[0], 140):
            for x in range(0, image.shape[1],140):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    	if len(boxes) == 0:
        	return []
    # initialize the list of picked indexes
    	pick = []

        # grab the coordinates of the bounding boxes
    	x1 = boxes[:,0]
    	y1 = boxes[:,1]
    	x2 = boxes[:,2]
    	y2 = boxes[:,3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:

            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]

            for pos in xrange(0, last):

                # grab the current index
                j = idxs[pos]

                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])

                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                overlap = float(w * h) / area[j]

                if overlap > overlapThresh:
                    suppress.append(pos)

            idxs = np.delete(idxs, suppress)

    	print boxes[pick]
        return boxes[pick]

    def load_model(window_image):
        des=window_image
        des=des.reshape(1,224,224,3)
        layer_name = 'fc1'
        intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(des)
        y_proba = clf.predict_proba(intermediate_output) ##
    	class_lab=clf.predict(intermediate_output)
        return y_proba

    global count_flowers
    global clone1
    count_flowers=0

    def show_window():
        global image_flowers
        global image_leaves
        global clone1
    	global file
        print ('Starting the image! ' + file + ' ' + str(datetime.datetime.now()))
        clone1=resized
        rowData = []
        rowData.append(file)

        for(x,y, window) in sliding_window(resized, 120, (wind_row,wind_col)):

            global count_flowers
            global image_flowers
            global image_leaves
            global clone1

            if window is None:
               continue
            if window.shape[0] != wind_row or window.shape[1] != wind_col:
               continue

            t_img = window
            img_1 = image_to_feature_vector(t_img,(224,224)) #should be (150528)
            img_data = np.array(img_1) # converted to array
            img_data = img_data.astype('float32') # converted to float
            img_data_scaled = preprocessing.scale(img_data) #preprocessed in life
            img_data=img_data_scaled
            img_data=img_data.reshape(1,224,224,3) # changed to (1,224,224,3)
            #8. Then, pass it to the CNN and get its feature of size 4096.
            #9. Pass this 4096 to the svm
            prediction =load_model(img_data)
            global coord

            if (prediction[0][1]> prediction[0][0]): #Applying threshold
                coord.append((x,y,x+140,y+140))
                rowData.append(prediction[0][1])
        return rowData
    
    start = timer()
    rowAsList = show_window()
    end = timer()
    
    print('Time for this image: '+str(end-start)+' secs')
    rowAsSeries = pd.Series(rowAsList)
    coord_array=np.array(coord)
    pick = non_max_suppression_slow(coord_array, 0.5)
    for xyxy in pick:
      cv2.rectangle(clone1, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (15,15,255), 9)
    
    length=len(pick)
    length=count_flowers
    print file
    print length
    df = df.append(rowAsSeries,ignore_index=True)
    name=file
    cv2.imwrite(path_for_output_images+"/"+name[:-4]+"_boxed.jpg", clone1)
    fileName = sys.argv[2]
    writer = pd.ExcelWriter(fileName, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
