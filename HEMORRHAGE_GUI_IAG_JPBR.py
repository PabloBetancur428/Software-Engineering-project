import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import models, transforms 
from torchvision.models import  ResNet50_Weights

class ImageClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")

        # Load the pretrained model
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # Create labels and buttons
        self.label = tk.Label(self.root, text="Select an image for classification:")
        self.label.pack(pady=10)

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.browse_button = tk.Button(self.root, text="Browse", command=self.load_image)
        self.browse_button.pack(pady=10)

        self.classify_button = tk.Button(self.root, text="Classify", command=self.classify_image)
        self.classify_button.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            self.image = Image.open(file_path)
            self.image = self.image.resize((224, 224))  # Resize image to match the model's expected size

            # Display the image on the GUI
            photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def classify_image(self):
        if hasattr(self, 'image'):
            # Preprocess the image
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(self.image)
            input_batch = input_tensor.unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                output = self.model(input_batch)

            # Get the predicted class
            _, predicted_idx = torch.max(output, 1)
            class_label = str(predicted_idx.item())

            # Save the result to a text file
            with open("classification_result.txt", "w") as file:
                file.write(f"Predicted Class: {class_label}")
        else:
            print("Please select an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierGUI(root)
    root.mainloop()