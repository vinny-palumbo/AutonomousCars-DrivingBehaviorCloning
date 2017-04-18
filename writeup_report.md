**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./writeup-images/center-lane-driving.png "Center Lane Driving"
[image2]: ./writeup-images/original-image.jpg "Original Image"
[image3]: ./writeup-images/flipped-image.jpg "Flipped Image"
[image4]: ./writeup-images/left_camera.jpg "Left Camera"
[image5]: ./writeup-images/center_camera.jpg "Center Camera"
[image6]: ./writeup-images/right_camera.jpg "Right Camera"
[image7]: ./writeup-images/cropped-image.jpg "Cropped Image"
[image8]: ./writeup-images/recovery-step0.png "Recovery Step 0"
[image9]: ./writeup-images/recovery-step1.png "Recovery Step 1"
[image10]: ./writeup-images/recovery-step2.png "Recovery Step 2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video-track1.mp4 containing a video of my model driving 1 full lap around track 1
* video-track2.mp4 containing a video of my model driving 1 full lap around track 2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5 convolutional layers (three 5x5 and two 3x3 filters with depths between 24 and 64) and 4 fully-connected layers (model.py lines 52-69) 

The model includes RELU layers to introduce nonlinearity (model.py lines 58-62), and the data is normalized in the model using a Keras lambda layer (model.py line 55). 

#### 2. Attempts to reduce overfitting in the model

The model contains 1 dropout layer after the first fully-connected layer in order to reduce overfitting (model.py line 65).
The model was trained and validated on different data sets to ensure that the model was not overfitting: 80% training data, 20% cross-validation data (model.py line 72). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 71).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of: 
* Center lane driving going a full lap in the opposite direction on Track 1 
* Center lane driving going a half-a-lap in the opposite direction on Track 2
* Recovering from the left and right sides of the road a dozen times on Track 1

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a fully-connected neural network with no hidden layer, to get started quickly. In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation set (20%). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

I then used a convolution neural network with the LeNet-5 architecture. I thought this model might be appropriate because it's widely used in Computer Vision and I had used it the the Traffic Sign Classifier project.
To combat the overfitting, I modified the model by adding a dropout layer. I then found out that I had a high mean squared error on the training set and on the validation set. This implied that the model with LeNet architecture was underfitting.
I tested the LeNet model in the simulator and the car fell off track 1 after a few seconds.

Then, I used the [Nvidia architecture](https://arxiv.org/pdf/1604.07316.pdf) suggested in the lesson for this project. The mean squared error on the training set and on the validation set were now both low.
I tested the Nvidia model in the simulator and the car successfully completed track 1 wihtout leaving the road, but couldn't complete track 2. 

To improve the driving behavior on track 2, I recorded half-a-lap going in the opposite direction on track 2. The model was now completing track 2, but couldn't complete track 1 like before.

Finally, I recorded the vehicle recovering from the left and right sides of the road back to center a dozen of times on track 1 only, so that the vehicle would learn to recover to the center rather than exit the road.
At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 52-69) consisted of a convolution neural network with the following layers

| Layer         	|     Description            			| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 160x320x3 normalized image   			| 
| Cropping 		|  65x320x3 cropped image   			| 
| Convolution 5x5     	| 2x2 stride, filter depth of 24		|
| RELU			|						|
| Convolution 5x5     	| 2x2 stride, filter depth of 36		|
| RELU			|						|
| Convolution 5x5     	| 2x2 stride, filter depth of 48		|
| RELU			|						|
| Convolution 3x3     	| filter depth of 64				|
| RELU			|						|
| Convolution 3x3     	| filter depth of 64				|
| RELU			|						|
| Flatten		| outputs 576	        			|
| Dropout		| 90%						|
| Fully connected	| outputs 250        				|
| Fully connected	| outputs 100        				|
| Fully connected	| outputs 50        				|
| Fully connected	| outputs 10        				|
| Fully connected	| outputs 1        				|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one full lap on track 1, going in the opposite directions and using center lane driving. 
Here is an example image of center lane driving:

![alt text][image1]

I trained my inital LeNet-5 architecture on this training dataset, which failed to complete track 1.

To augment the dataset, I then flipped all images and inversed corresponding steering angles thinking that this would remove the right-turn bias from Track 1 in the opposite direction (model.py lines 36-42).
Here is an example image that has then been flipped:

![alt text][image2]
![alt text][image3]

My LeNet-5 architecture was still failing to complete track 1.

To augment the dataset even more, I used the images captured by the left and right cameras as well and set their respective steering angle to the center image steering angle plus or minus a correction of 0.2 (model.py lines 12-34).
Here are example images taken simultaneously from the left, center and right cameras respectively:

![alt text][image4]
![alt text][image5]
![alt text][image6]

My LeNet-5 architecture was still failing to complete track 1.

To remove noise (sky, trees, car hood, etc.) from the top and bottom pixels of the images, I cropped 70 pixels from the top and 25 pixels from the bottom of all images passed through the model (model.py line 57).
Here is an example image before and after being cropped:

![alt text][image2]

![alt text][image7]

My LeNet-5 architecture was still failing to complete track 1.

I decided to change my convolutional neural network architecture from LeNet to [this one from Nvidia](https://arxiv.org/pdf/1604.07316.pdf).
The new architecture finally succeeded to complete track 1, but failed to complete track 2. 

I then added to the training dataset half-a-lap of track 2, going in the opposite direction again.
This new training data from track 2 was also subject to the same training process steps mentionned above: center-lane driving, flipping images, using all 3 cameras, cropping top and bottom pixels.
The Nvidia architecture finally succeeded to complete track 2, but was now failing to complete track 1. 

Finally, I recorded the vehicle recovering from the left and right sides of the road back to center a dozen of times on track 1 only, so that the vehicle would learn to recover to the center rather than exit the road.
These images show what a recovery looks like :

![alt text][image8]
![alt text][image9]
![alt text][image10]

The Nvidia architecture finally succeeded to complete both track 1 and 2. 

With the final collection process, I had a training dataset of 12,786 images and steering angles.
