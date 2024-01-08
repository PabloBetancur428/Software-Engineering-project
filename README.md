# Software-Engineering-project
The primary objective of this code was to develop a straightforward Graphical User Interface (GUI) capable of categorizing images of CT hemorrhages and normal CT scans of the brain. The GUI was constructed using Tkinter and TensorFlow, providing the ability to browse for images within any folder on the current PC. The pretrained model, accessible on GitHub (URL provided),incorporates multiple convolutional layers, alongside Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) components. After testing, the pretrained model achieved a commendable accuracy of 88.33%. 

In this repository you will find these files:
1) Hemorrhage_ModelPred_IAG_JPBR.ipynb : In this file the model was pretrained in a googlecolab environment in order to have access to good quality GPU's. You can reproduce this colab notebook just by creating the necessary folder in your drive with the information from the dataset (The URL for that is also included inside the notebook).
   
3) Hemorrhage_GUI_IAG-JPBR.py : Here, you will find the GUI created with tkinter. All you have to do is change the path where your model is stored and it will be ready to go as long as you have the needed libraries like tensorflow version 2.15.0, tkinter version 8.6 and pillow version 10.1.0. This model is able to read images in .png, .jpg, .jpeg format.
   
5) traning_model.h5 : This file is the pretrained model in (1). That is a keras CNN, so in order to read it and keep the numerical values of the weights, is recomendable to read it with tensorflow if you don't want to lose any information.
   
7) Test: This folder contains some test images from the original dataset that you can use to test the model. You will find two notations: 'h' stands for hemorrhage image and 'n' that stands for normal image.
