# Face-Mask-Detection-Deep-Neural-Network-Triple-Solution
Three types of solutions to the Mask-Detection problem where the first is presented in the form of a convolution network, the second in a network that is fully connected and the third is based on a transfer network. All of them are based Tensorflow keras.
<ul>
  <li>
    <h2>Background</h2>
    This report describes the mask detection problem and its solution. It contains three configurations that describe three different neural networks: the first, a Fully Connected Network. The second, transfer-learning based network and the third, a Convolutional neural network. 
In this report you will find the complete process of solving this problem using the models described above and additional information such as graphs, code lines screenshots, Loss vs Validation, and other ML concepts that will contribute to the understanding of our project.

  </li>
  
  <li>
    <h2>Data Description</h2>
    The data used for training the model contains 10,000 images which half of them shows a person with a face-mask and the other half non-masked faces.The data we are using contains a validation set, with 1000 pictures, 500 masked-faces and 500 non-masked faces.
The third set, used for testing the model, contains 1000 pictures, 500 masked-faces and 500 non-masked faces.
The data you can get for free from www.kaggle.com from the link <a href="https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
">Here</a>
  </li>
  
   <li>
    <h2>Networks and Architecture</h2>
    <b>The first configuration</b> was built by us from end to end, the architecture used in this configuration is <li>
      CNN- Convolutional Neural Network which consists of several layers that implement feature extraction, and then classification.
The CNN layers:
Conv2D Layer - The filter parameter means the number of this layer's output filters
The kernal_size parameter is commonly used 3*3.
The activation parameter refers to the type of activation function.
The padding parameter is enabled to zero-padding.
The input_shape parameter has pixel high and pixel wide and have the 3 color channels: RGB
</li>
<li>
MaxPool2D Layer - to pool and reduce the dimensionality of the data.
 </li>
 <li>
Flatten Layer -  flatten is used to flatten the input to a 1D vector then passed to dense.
  </li>
  <li>
Dense Layer (The output layer) - the units parameter means that it has 2 nodes one for with and one for without because we want a binary output.
The activation parameter - we use the RELU and SIGMOID activation functions on our output so that the output for each sample is a probability distribution over the outputs of with and without mask.
  </li> 
   <br> <img src="report_images/unnamed.jpg"><br>
   <b>The second configuration</b> is a transfer-learning configuration, in that case we used the MobileNet V2 architecture. 
MobileNetV2 is a convolutional neural network architecture that seeks to perform well on mobile devices. 
It is based on an inverted residual structure where the residual connections are between the bottleneck layers.
We fine-tuned MobileNetV2 on our mask/no mask dataset and obtained a classifier.
In MobileNetV2, there are two types of blocks:
One is a residual block with stride of 1. Another one is a block with stride of 2 for downsizing.
There are 3 layers for both types of blocks.This time, the first layer is 1×1 convolution with ReLU6.	The second layer is the depthwise convolution.
The third layer is another 1×1 convolution but without any non-linearity. It is claimed that if RELU is used again, the deep networks only have the power of a linear classifier on the non-zero volume part of the output domain.to the network we imported we added a few fully connected layers of our own.
  <br> <img src="report_images/mobile.JPG"> <br>
  <b>The third configuration</b> we used is Fully Connected Network which consists of several fully connected hidden layers. 
fully connected layer is a function from ℝ m to ℝ n. Each output dimension depends on each input dimension. 
The Fully Connected Neural Network layers:
<li>
Flatten Layer -  flatten is used to flatten the input to a 1D vector then passed to dense.
 </li>
 <li>
Dense Layer - we have two dense layers, in the first we used the RELU activation function and in the second Dense layer (the output layer)  we used the  Sigmoid activation function.
  </li>
   <br> <img src="report_images/fcn.png"><br>

   
  </li>
  
   <li>
    <h2>Training Process</h2>
    example example example
  </li>
  
   <li>
    <h2>Conclusion</h2>
    example example example
  </li>
</ul>
