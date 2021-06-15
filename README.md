# mask_detector

This is my first ML project. Wish me luck.

My aim is to create a simple python script that open live camera then giving a sign wether the mask is being used or not. My plan is to use tensorflow model and opencv to predict and detect faces. Perhaps I could put my work into a website so that people can try what I have done.

So far I have created
- Machine Learning model that can detect people wearing their mask or not. I am using EfficientNetB7 mainly for the convolutional part of my model. And I am getting a pretty good accuracy with the model. But the size is too big and the parameters are to many. I also opted to use mobilenetv2 if I needed smaller model. I create the model using google colab.
- OpenCV script that can detect faces. I am using Haar Cascade method to detect faces.

To Do List:
- Extracting the face information from OpenCV then fed it into the model to get the prediction wether they are wearing their mask or not.