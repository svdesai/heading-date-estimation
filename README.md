# flowering-panicle-detection

notes : 

- include parts of abstract to give a context

Introduction: 

This is an implementation of a yet to be submitted paper titled 'To Go Deep or Not To: Detection of Flowering Panicles in Paddy Rice'. To plan the perfect time for harvest of rice crops, we detect and quantify the flowering of paddy rice. The dataset of rice crop images used in this work is taken from [1]. 

- brief description of the algorithms used

Methodology : 

The basic outline of the method is as follows. We use an SVM classifier with a sliding window to detect flowering panicles in the image. To generate the features to be given as input to the SVM classifier, we use two methods of feature extraction and compare their performances. 

 - Feature extraction using SIFT Descriptors
 - Feature extraction using a Deep neural network (VGG-16 [2])

Training is done using patches sampled from 21 images of Kinmaze dataset. Testing is done on the full 5184x3456 images from Kinmaze dataset. [1]. 

- How to Run
- Results

The ROC curves obtained for the SVM classifier after performing cross validation on training dataset are as follows : 

[image 1]

[image 2]

The results obtained on the test set are as follows : 

[Table 1]


Acknowledgements: Paper written under the able guidance of 

 - Dr Vineeth N Balasubramanian
 - Dr Wei Guo 

- Code Contributors

1. Manasa Kumar [github]
2. Sai Vikas [github]


References : 

[1] Guo et al.'s paper
[2] VGG Paper
