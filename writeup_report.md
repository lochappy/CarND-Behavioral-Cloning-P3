# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figures/driveNet.png "Model Visualization"
[image2]: ./figures/center_2017_03_23_22_53_22_491.jpg
[image3]: ./figures/flipped_center_2017_03_23_22_53_22_491.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes with depths between 24 and 64 (model.py lines 58-64) 

The model includes parametreic RELU layers to introduce nonlinearity (model.py line 56), and the data is normalized in the model using a Keras lambda layer (model.py line 47), and cropped by 50 top and 25 bottom rows (model.py line 50). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 80). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 181-183). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 171).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I only used center lane driving images.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from a well-known architecture like LeNet, then the [NVIDIA Drive Net](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), finnally came out with the modified [NVIDIA Drive Net](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

My first step was to use a convolution neural network model similar to the LeNet (model.py line 14) I thought this model might be appropriate because it performed very well in the traffic sign classification project. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The validation loss of my first model saturated at 0.0130~0.0145, no matter how much data I pumped to the training. When this model was used in the autonomous mode of the simulation, the car behaved well in the straight road. However, it behaved terribly at the curved road, easily went off the road. This implied that this model did not capture the turning corners well.

Then I tried the more poweful model [NVIDIA Drive Net](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) as my second trial. The second model behaved pretty well at the curved road. However, it easily went off the road at the sharped turning corners.

My third model is the modified version of the second one. This model used the parametric RELU instead of the normal RELU and an additional DROPOUT layer with ratio 0.5. This model dealed with the sharped turning corners pretty well. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 42-88) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						|
| Normalization         | Scale and center input to [-1,1]				|
| Cropping              | Top 50, bottom 25, output 85x320x3			| 
| Conv1 5x5     	    | 1x1 stride, valid padding, outputs 81x316x24 	|
| PRELU					|												|
| Max pooling       	| 2x2 stride, valid padding, outputs 40x158x24 	|
| Conv2 3x3     	    | 1x1 stride, valid padding, outputs 38x156x36 	|
| PRELU					|												|
| Max pooling       	| 2x2 stride, valid padding, outputs 19x78x36 	|
| Conv3 3x3     	    | 1x1 stride, valid padding, outputs 17x76x48 	|
| PRELU					|												|
| Max pooling       	| 2x2 stride, valid padding, outputs 8x38x48 	|
| Conv4 3x3     	    | 1x1 stride, valid padding, outputs 6x36x64 	|
| PRELU					|												|
| Conv5 3x3     	    | 1x1 stride, valid padding, outputs 4x34x64 	|
| PRELU					|												|
| Flatten	            | outputs 8704     								|
| Fully connected 	    | outputs 100       							|
| Fully connected 	    | outputs 50       							    |
| Dropout 	            | Ratio 0.5, outputs 50       					|
| Fully connected 	    | outputs 10       							    |
| Fully connected 	    | outputs 1       							    |

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I have recorded more than 20 laps on track one. In order to diverse the training data, I have also recorded the same amount of data in the reversed driving direction. The total number of training images are 35000 data points.Here is an example image of center lane driving:

![alt text][image2]

To augment the data set, I also flipped images and angles thinking that this would add more variation to the training data, making the model more robust, resulting in more then 70000 data points. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image3]

I finally randomly shuffled the data set and put 20% (model.py line 179) of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I also set up a model checkpoint (model.py line 179), which helped me navigate the training procedure, and save the model that had the lowest error on the valiadtion set. I used an adam optimizer so that manually training the learning rate wasn't necessary.
