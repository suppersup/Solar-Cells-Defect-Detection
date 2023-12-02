# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'basic_layout.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

#Libaries for UI
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import os

# Libaries for model
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from numpy import genfromtxt
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]

        if self.transform:
            images = self.transform(images)

        # Ensure that the image is a PyTorch tensor and the probability is a float tensor
        images = torch.Tensor(images)
        
        return images

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts images to PyTorch tensors
    transforms.Resize((300, 300)),
])

def predict_function(model, images_array, device='cpu'):    
    images_dataset = CustomDataset(images_array, transform=transform)
    images_loader = DataLoader(images_dataset, batch_size=1, shuffle=False)

    # Make sure the model is on the same device as the input tensor
    model = model.to(device)
    predictions = []

    with torch.no_grad():
        for images in images_loader:
            images = images.to(device)
            outputs = model(images)
            print(outputs)
            predictions.append(outputs.item())

    return predictions

            # Use outputs as needed
     
# Define the CNN model with L2 regularization For One Channel
class OneChannel_CNN_Probability(nn.Module):
    def __init__(self):
        super(OneChannel_CNN_Probability, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # One channel input
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # Adjust the input size for FC1
        self.fc1 = nn.Linear(256 * 37 * 37, 256)  # Calculate the input size based on the convolutions and pooling
        self.fc2 = nn.Linear(256, 1)  # One output unit for binary classification with sigmoid activation

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary classification
        return x
    
#main UI window
class Ui_mainWindow(object):
    
    loaded_images = None
    model = None
    folder_path = None
    predictions = None

    #define the UI
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(1600, 900)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Create a scroll area
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setWidgetResizable(True)

        # Create a widget to hold the Matplotlib figure
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 700, 500))

        # Create a vertical layout for the widget
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        # Create a Matplotlib figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.verticalLayout_2.addWidget(self.canvas)

        # Set the widget to the scroll area
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        # Create a vertical layout for the central widget
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        # Add the scroll area to the layout
        self.verticalLayout_3.addWidget(self.scrollArea)

        # Create QPushButton widgets
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")

        # Add the buttons to the layout
        self.verticalLayout_3.addWidget(self.pushButton)
        self.verticalLayout_3.addWidget(self.pushButton_2)
        self.verticalLayout_3.addWidget(self.pushButton_3)

        mainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 721, 21))
        self.menubar.setObjectName("menubar")
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)
        self.pushButton.clicked.connect(self.load_file)
        self.pushButton_2.clicked.connect(self.load_model)
        self.pushButton_3.clicked.connect(self.predict)

        # Create a subplot in the Matplotlib figure
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')

#define all the functions
    def load_file(self):
        global loaded_images
        global folder_path
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(mainWindow, 'Select Folder')
        Ui_mainWindow.folder_path = folder_path

        if folder_path:
            print(f"Selected folder: {folder_path}")
            
            self.loaded_images = self.load_images(folder_path)
            print(self.loaded_images.shape)
            self.displayImages(folder_path)
        
    def load_images(self, folder_path):
        images = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif')

        try:
        # List all files in the directory and filter out those with null characters
            filenames = [f for f in os.listdir(folder_path) if '\x00' not in f]
        
            for filename in filenames:
                if filename.lower().endswith(valid_extensions):
                    img = mpimg.imread(os.path.join(folder_path, filename))
                    images.append(img)

        except Exception as e:
            print(f"Error loading images: {e}")

        images_array = np.array(images)
        Ui_mainWindow.loaded_images = images_array

        return images_array

    def load_model(self):
        model_path, _ = QtWidgets.QFileDialog.getOpenFileName(mainWindow, 'Select File', '', 'Models (*.pth);;All Files (*)')
        if model_path:
            print(f"Selected file: {model_path}")
            try:
                checkpoint = torch.load(model_path)
                if isinstance(checkpoint, OneChannel_CNN_Probability):
                # If the checkpoint is an instance of OneChannel_CNN_Probability, use it directly
                   self.model = checkpoint
                else:
                # If it's not an instance, assume the entire model is saved and create an instance
                    self.model = OneChannel_CNN_Probability()
                    self.model.load_state_dict(checkpoint)
                
                self.model.eval()
            except Exception as e:
                print(f"Error loading the model: {e}")

        Ui_mainWindow.model = self.model
    
    def predict(self):
        if Ui_mainWindow.model is not None and Ui_mainWindow.loaded_images is not None:
            predictions = predict_function(Ui_mainWindow.model, Ui_mainWindow.loaded_images)
            Ui_mainWindow.predictions = predictions  # Store predictions as an attribute
            self.displayImages(Ui_mainWindow.folder_path, Ui_mainWindow.predictions)

        else:
            print("Model or images not loaded.")  

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "I CAN TELL THE DEFECTS!! :D"))
        self.pushButton.setText(_translate("mainWindow", "WHERE IS THE IMAGE!! :D"))
        self.pushButton_2.setText(_translate("mainWindow", "WHERE IS MY MODEL!! x("))
        self.pushButton_3.setText(_translate("mainWindow", "Predict MAMAMIYA :D"))

    def displayImages(self, folder_path, predictions=None):
    # Clear the existing subplots
        self.figure.clear()
        self.ax.clear()
        self.ax.axis('off')

    # Create a grid layout for subplots
        grid_layout = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)

    # Counter for keeping track of the position in the grid
        row = 0
        col = 0

    # Iterate through files in the selected folder
        images_to_plot = 100  # Set the number of images to plot
        for i, filename in enumerate(os.listdir(folder_path)):
            if i >= images_to_plot:
                break

            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Read the image using Matplotlib
                img = mpimg.imread(os.path.join(folder_path, filename))

            # Create a new subplot for each image
                ax = self.figure.add_subplot(10, 10, i + 1)
                ax.imshow(img)
                ax.axis('off')

            # Display the filename below the image
                ax.text(0.5, -0.15, f"{filename}", size=8, ha="center", transform=ax.transAxes)

            # Display the prediction if available
                if predictions is not None:
                    prediction = predictions[i]
                    ax.set_title(f"% Defect: {prediction*100:.2f}", size=8)


            # Increment row and column for the next iteration
                col += 1
                if col > 9:
                    col = 0
                    row += 1

    # Adjust layout
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())





    

