
## Introduction

This is an implementation of submitted paper: 

Desai, S. V., Balasubramanian, V. N., Fukatsu, T., Ninomiya, S., & Guo, W. (2019). Automatic estimation of heading date of paddy rice using deep learning. Plant Methods, 15(1), 76. https://doi.org/10.1186/s13007-019-0457-1 

To plan the perfect time for harvest of rice crops, we detect and quantify the flowering of paddy rice. The dataset of rice crop images used in this work is taken from [1].

## Methodology 

The basic outline of the method is as follows. We use an SVM classifier with a sliding window to detect flowering panicles in the image. To generate the features to be given as input to the SVM classifier, we use two methods of feature extraction and compare their performances.
 - Feature extraction using SIFT Descriptors
 - Feature extraction using a deep neural network (VGG-16 [2])

Training is done using patches sampled from 21 images of Kinmaze dataset. Testing is done on the full 5184x3456 images from Kinmaze dataset. [1].

## How to Run

To obtain training accuracy : 

```bash

python training_accuracy/sift/sift_training_accuracy.py

python training_accuracy/vgg/vgg_training_accuracy.py

```

To obtain correlation : 

```bash
python correlation/kinmaze_data/vgg/vgg_testing_on_kinmaze.py path_to_directory_containing_test_images/ output_probs.xlsx
```

## Results

The ROC curves obtained for the SVM classifier after performing cross validation on patches are as follows :



| SIFT Features with D=5 Pixels | VGG Features          | 
| ------------- |:-------------:| 
| <img src="https://i.imgur.com/ibPYUpn.png" width="400" />    | <img src="https://i.imgur.com/QWnS4EK.png" width="450" /> |

Flowering panicles detected in one of the images of the Kinmaze dataset: 

<img src="https://i.imgur.com/AaI9BCf.jpg" width="800" />

The correlation results obtained on the whole-images are as follows :

| Approach | Pearson Correlation Coefficient          | 
| ------------- |:-------------:| 
| VGG     | **0.7495** | 
| SIFT      | 0.5435       | 

The computational cost for VGG based method is as follows : 

| Task | Time Taken          | 
| ------------- |:-------------:| 
| Feature extraction using VGG and training the SVM classifier     | 42 seconds | 
| Detecting flowering panicles      | 29 seconds per image       |

GPU Used : NVIDIA Geforce GTX 1080Ti (11 GB Memory)

## Acknowledgements 
Paper written under the able guidance of :
 
  **Dr. Vineeth N Balasubramanian**,  IIT Hyderabad, India.

  **Dr. Wei Guo**, University of Tokyo, Tokyo.

## Code Contributors
Manasa Kumar [[github]](https://www.github.com/manasaKay/)

Sai Vikas [[github]](https://www.github.com/saivikas3/)

## References
[1] Guo W, Fukatsu T, Ninomiya S. *Automated characterization of flowering dynamics in rice using field-acquired time-series RGB images.* Plant Methods. 2015;11:7. [doi:10.1186/s13007-015-0047-9](https://doi.org/10.1186/s13007-015-0047-9)

[2] Karen Simonyan, Andrew Zisserman. *Very Deep Convolutional Networks for Large-Scale Image Recognition.* [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)

