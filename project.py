# The primary objective of this code was to develop a straightforward Graphical User Interface (GUI) 
# capable of categorizing images of CT hemorrhages and normal CT scans of the brain. 
# The GUI was constructed using Tkinter and TensorFlow, providing the ability to browse 
# for images within any folder on the current PC. The pretrained model, accessible on GitHub (URL provided),
# incorporates multiple convolutional layers, alongside Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) components.
# After testing, the pretrained model achieved a commendable accuracy of 88.33%. 

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import models, transforms
import tensorflow as tf
from tensorflow.keras.preprocessing import image

#First, let's load the model that is going to be used in this project
model_path = 'C:/Users/Angela Rengifo/Desktop/RoadMap/Projects/New_course_OD/trading_model.h5'
loaded_model = tf.keras.models.load_model(model_path)


class ImageClassifierGUI:
    def __init__(self, root):
        #Initialize the tkinter window
        self.root = root 
        #self.root.geometry('250x300')
        self.root.title("BRAIN HEMORRHAGES")
        self.gui_widgets()

    def gui_widgets(self):
        #Create a template with the main characteristics of the main window

        self.label = tk.Label(self.root, text = "Select a CT scan image for classification")
        self.label.pack(pady = 10)

        self.message_label = self.message_label = tk.Label(self.root)
        self.message_label.place(x = 30, y = 50)
    

        self.label2 = tk.Label(self.root)
        self.label2.pack(pady = 20)

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.upload_img = tk.Button(self.root, text= "Browse", command = self.load_img)
        self.upload_img.pack()

        self.classify_b = tk.Button(self.root, text = 'Classify', command = self.classify_img)
        self.classify_b.pack()

    def load_img(self):

        #Browse image file
        file_path = filedialog.askopenfilename(filetypes = [("Image files", "*.png;*.jpg;*.jpeg")])
        
        
        if file_path:
            self.image_l = Image.open(file_path)
            self.image_l = self.image_l.resize((256,256)) # Resize image to the trained model shape size

            #Display image in GUI
            photo = ImageTk.PhotoImage(self.image_l) #convert the loaded image to PhotoImage object in order to display it on the screen
                                                        
            self.image_label.config(image = photo) #Update label that was empty with the uploaded image
            self.image_label.image = photo

            #Convert image to array so it can be feeded to the prediction model
            self.image_test = image.load_img(file_path, target_size=(256, 256), color_mode='grayscale')
            self.img_array = image.img_to_array(self.image_test)/255.0
            self.Array_Image = self.img_array.reshape((1,) + self.img_array.shape)

    def classify_img(self):

        #Load the model
        self.loaded_model = tf.keras.models.load_model(model_path)

        #Make a prediction and take the maximum argument so it gives a 1 for hemorrhage and 0 for normal
        self.prediction = loaded_model.predict(self.Array_Image)
        self.prediction = self.prediction.argmax(axis = -1)

        self.message_label['text'] = ''
        
        #Display a message according to the prediction
        if self.prediction[0] == 0:

            self.message_label = tk.Label(self.root, text="NO HEMORRHAGE", fg="green", font=("Arial", 12))
            self.message_label.place(x = 60, y = 50)

            
        else:
            
            self.message_label = tk.Label(self.root, text="HEMORRHAGE DETECTED!!!",fg="red", font=("Arial", 12))
            self.message_label.place(x = 25, y = 50)                

#Run the GUI
root = tk.Tk()
app = ImageClassifierGUI(root)
root.mainloop()
