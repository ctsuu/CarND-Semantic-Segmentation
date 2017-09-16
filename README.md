# Semantic Segmentation

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Overview

The main goal for this project is apply Fully Convolution Neural Network(FCN) idea to segragate the road surface from all other objects such as sidewalk, other cars, people, building, grass etc. FCN will take advantage of pretrained mode such as VGG 16 mode, use it as input, (the encoder part), then transfer to the decoder part, upsample to the same size as input image. There are 2 skipped connections in between encoder and decoder. This operation preserves the critical spatial information. 

KITTI dataset includes training set and test set. Orginal image size is 1242x375. We are resize to 576x160. The FCN pipeline is based on the new image size. Each training image pair with ground true image which shows marked road surface. FCN trained on this dataset, test on unseed test dataset. 

### Pretrained VGG Model

Train a Convolution Neural Network is time and resource consuming process. The good new is there are some good pretrained model available to accelerate the process. The choosen VGG 16 has 7 convolution layers, we can export input layer, the VGG drop out rate, the layer 3, 4, 7 out for further process. 

The VGG model is available at https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip.  

### Encoder and Decoder




### Optimize and Train the Model

### Model Performance

### Future Works


### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder
 
 ## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
