import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import models, transforms
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the pre-trained model for image classification
model_path = 'C:/Users/Angela Rengifo/Desktop/RoadMap/Projects/New_course_OD/traning_model.h5'
loaded_model = tf.keras.models.load_model(model_path)

class ImageClassifierGUI:
    def __init__(self, root):
        # Initialize the tkinter window
        self.root = root 
        self.root.title("BRAIN HEMORRHAGES")
        self.gui_widgets()

    def gui_widgets(self):
        # Create GUI elements

        # Label to instruct the user to select an image
        self.label = tk.Label(self.root, text="Select a CT scan image for classification")
        self.label.pack(pady=10)

        # Label to display messages
        self.message_label = tk.Label(self.root)
        self.message_label.place(x=30, y=50)

        # Placeholder label for displaying an image
        self.label2 = tk.Label(self.root)
        self.label2.pack(pady=20)

        # Label to display the selected image
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # Button to browse and load an image
        self.upload_img = tk.Button(self.root, text="Browse", command=self.load_img)
        self.upload_img.pack()

        # Button to classify the loaded image
        self.classify_b = tk.Button(self.root, text='Classify', command=self.classify_img)
        self.classify_b.pack()

    def load_img(self):
        # Browse image file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            # Load and resize the image
            self.image_l = Image.open(file_path)
            self.image_l = self.image_l.resize((256, 256))

            # Display the image in the GUI
            photo = ImageTk.PhotoImage(self.image_l)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Convert the image to an array for model input
            self.image_test = image.load_img(file_path, target_size=(256, 256), color_mode='grayscale')
            self.img_array = image.img_to_array(self.image_test) / 255.0
            self.Array_Image = self.img_array.reshape((1,) + self.img_array.shape)

    def classify_img(self):
        # Load the model
        self.loaded_model = tf.keras.models.load_model(model_path)

        # Make a prediction and determine the class
        self.prediction = loaded_model.predict(self.Array_Image)
        self.prediction = self.prediction.argmax(axis=-1)

        # Clear previous messages
        self.message_label['text'] = ''

        # Display a message based on the prediction
        if self.prediction[0] == 0:
            self.message_label = tk.Label(self.root, text="NO HEMORRHAGE", fg="green", font=("Arial", 12))
            self.message_label.place(x=60, y=50)
        else:
            self.message_label = tk.Label(self.root, text="HEMORRHAGE DETECTED!!!", fg="red", font=("Arial", 12))
            self.message_label.place(x=25, y=50)

# Run the GUI
root = tk.Tk()
app = ImageClassifierGUI(root)
root.mainloop()
