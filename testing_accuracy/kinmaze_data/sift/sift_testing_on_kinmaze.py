#===IMPORTS===##

from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
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
from sklearn.metrics import roc_curve, auc
from feature_aggregation import BagOfWords

from sklearn.calibration import CalibratedClassifierCV
import time
import os,cv2
import numpy as np
import random
import pandas as pd
#custom modules
from bow_model_training import sift, dsift, image_to_feature_vector
from bow_model_testing import sliding_window, non_max_suppression_slow


##==================================PARAMETERS================================================##
stepSize=140
wind_row, wind_col = 140,140
img_rows, img_cols = 140,140
k=800

##Seeding for Random Numbers
seed=7 #used below for train/test split
np.random.seed(seed)
img_data_list=[]
num_channel=1

##FILE PATHS
trainPath = 'new_full_train_140' #Training patches
testPath = 'kinmaze_original'  ##Add the path to the test images here

#===================================TRAINING CODE===============================================##

##Adding images from training directory to list
listTrainImages=sorted(os.listdir(trainPath))
print len(listTrainImages)

#sanity checks : Count the number of flower and leaf patches
areFlowers = True
areLeaves  = True

for i in range(0,512):
    if listTrainImages[i][0] != 'f':
      areFlowers = False

for i in range(512,1572):
    if listTrainImages[i][0] != 'l':
       areLeaves = False

print areFlowers,areLeaves

for file in listTrainImages:
    input_img=cv2.imread(trainPath+'/'+file,0)
    # resize to 140x140 and get feature vector
    input_img_flatten=image_to_feature_vector(input_img,(img_rows,img_cols))
    img_data_list.append(input_img_flatten) #append all feature vectors to img_data_list

##Now that we have the feature vectors of images in a list : img_data
##Let's prepare the samples
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
#standardize the image dataset (subtract mean and divide by standard deviation)
img_data_scaled = preprocessing.scale(img_data)
#change shape to (782,140,140,1)
img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
img_data=img_data_scaled


##Create the labels array
num_of_samples = img_data.shape[0] #782
labels = np.ones((num_of_samples,),dtype='int64')
labels[0:511]=1 #512 are flowering panicles, hence 1
labels[511:]=0  #remaining are leaves

#shuffle data and labels (together) while preserving data-label correspondence
x,y = shuffle(img_data,labels, random_state=2)
#split dataset into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
#shapes after splitting (update this if test_size is changed)
# X_train : (625,140,140,1)
# Y_train : (625,)
# X_test : (157,140,140,1)
# Y_test : (157,)
print 'finding out SIFT descriptors..'

##get sift descriptors for each datapoint(image) in X_train
features_train = [
    dsift((m.reshape(140,140,1)*255).astype(np.uint8))
    for m in X_train
]
#each datapoint in features_train has dimensions: (784,128)
#this is because: the size of image is 140x140. Step size = 5.
# 140/5 = 28. Total keypoints : 28x28 = 784.
# 128 bin values for each keypoint. Therefore, (784,128)


bow_train = BagOfWords(800) # 800 is the number of codewords
bow_train.fit(features_train) #use k means clustering to map image patches to codewords
img_bow_train = bow_train.transform(features_train) #transform the histogram of codewords into an image

print 'fitting the svm model.. '
## train the SVM classifier with chi2 kernel
svm = SVC(probability=True,kernel=chi2_kernel).fit(img_bow_train,Y_train)


## get SIFT descriptors for datapoints in test set
features_test = [
    dsift((n.reshape(140,140,1)*255).astype(np.uint8))
    for n in X_test
]

print 'testing..'
#convert the features into images for testing
img_bow_test = bow_train.transform(features_test)
#test the SVM classifier and print the score
print svm.score(img_bow_test,Y_test)

##================================TRAINING CODE ENDS==================================================##

##===============================SLIDING WINDOW CODE==================================================##
def load_model(window_image):

    n=window_image
    #generate feature vectors
    features_test_here = [
    dsift((n.reshape(140,140,1)*255).astype(np.uint8))
    ]

    #convert the feature set into an image
    img_bow_test_here = bow_train.transform(features_test_here)

    #predict probabilities of each class
    y_proba = svm.predict_proba(img_bow_test_here)

    return y_proba


def show_window(resized):
    global image_flowers
    global image_leaves
    global clone1
    global file
    #threshold = 0.5  #if svm predicts flower with a confidence lesser than threshold, count won't be incremented.
    rowData = []
    rowData.append(file)
    for(x,y, window) in sliding_window(resized, stepSize, (wind_row,wind_col)):

        global count_flowers
        global image_flowers
        global image_leaves
        global clone1

        #if shape is not equal to the sliding window shape, continue
        if window.shape[0] != wind_row or window.shape[1] != wind_col:
            continue

        currentWindow = window

        currWindowFeatures = image_to_feature_vector(currentWindow,(140,140)) #should be (150528)
        img_data = np.array(currWindowFeatures) # converted to array
        img_data = img_data.astype('float32') # converted to float
        img_data_scaled = preprocessing.scale(img_data) #preprocessed in life (standardization)

        img_data=img_data_scaled
        img_data=img_data.reshape(1,140,140,1) # changed to (1,140,140,1)

        prediction=load_model(img_data)
        global coord

        if prediction[0][1]>prediction[0][0]: #if P(panicle exists) > P(panicle doesn't exist)
	       rowData.append(prediction[0][1])
            #if prediction[0][1]> threshold: # if the prediction is pretty confident
           coord.append((x,y,x+140,y+140))
            #    count_flowers+=1 #increment the flowering panicle count
    return rowData





df = pd.DataFrame({'Name': [],'Value':[]})

global k
k=0

#Do the sliding window test for each image in the directory : testPath
listTestImages=os.listdir(testPath)
listTestImages = sorted(listTestImages)
for file in listTestImages:

    print ('Starting the image!')

    global k
    global coord
    global count_flowers
    global clone1

    coord=[]
    img=cv2.imread(testPath+'/'+file,0)
    resized=img




    #set flower count to zero
    count_flowers=0
    rowAsList = show_window(resized) #the method which runs svm and detects flowers
    rowAsSeries = pd.Series(rowAsList)

    print rowAsList

    coord_array=np.array(coord) #the array of rectangles in which a flower exists

    pick = non_max_suppression_slow(coord_array, 0.5)

    #draw rectangles at each set of coordinates in the coord_array
    for (x1,x2,x3,x4) in coord_array:
        cv2.rectangle(resized, (x1, x2), (x3,x4), (15, 15, 255), 3)

    length=len(pick)

    print file
    print length
    print time.asctime( time.localtime(time.time()) )
    #df=df.append({'Name':file,'Value':length},ignore_index=True)

    df = df.append(rowAsSeries,ignore_index=True)


    #write the dataframe to a spreadsheet
    writer = pd.ExcelWriter('kinmaze_allprobs.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
