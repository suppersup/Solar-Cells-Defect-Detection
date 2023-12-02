# libarires
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


#check device 
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)
print(device)


#import data
from elpv_reader import load_dataset
images, proba, types = load_dataset()




print(type(proba))
print(type(images))
print(type(types))
print(proba.shape)
print(images.shape)
print(types.shape)

#split data
train_n = 0.8              #percentage of the training data set
test_n = train_n+((1-train_n)/2)     #percentage of the testing data set

#split data
# training data set 80% of the total data set
train_set = images[:int(len(images)*train_n)]
train_proba = proba[:int(len(proba)*train_n)]
train_types = types[:int(len(types)*train_n)]
# testing data set 10% of the total data set
test_set = images[int(len(images)*train_n):int(len(images)*test_n)]
test_proba = proba[int(len(proba)*train_n):int(len(proba)*test_n)]
test_types = types[int(len(types)*train_n):int(len(types)*test_n)]
# validation data set 10% of the total data set
val_set = images[int(len(images)*test_n):]
val_proba = proba[int(len(proba)*test_n):]
val_types = types[int(len(types)*test_n):]  




n = 100
sample_images = train_set[:n]
sample_proba = train_proba[:n]
sample_types = train_types[:n]

# Create a 10x10 grid of subplots for the first 100 images
fig, axes = plt.subplots(10, 10, figsize=(12, 12))
fig.suptitle("Sample Images with Proba and Type")

# Plot each image with proba and type
for i in range(n):
    ax = axes[i // 10, i % 10]
    ax.imshow(sample_images[i])
    ax.set_title(f"Proba: {sample_proba[i]:.2f}\nType: {sample_types[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()
class CustomDataset(Dataset):
    def __init__(self, images, probabilities, transform=None):
        self.images = images
        self.probabilities = probabilities
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]
        probabilities = self.probabilities[idx]

        if self.transform:
            images = self.transform(images)

        # Ensure that the image is a PyTorch tensor and the probability is a float tensor
        images = torch.Tensor(images)
        
        probability = torch.Tensor([probabilities])  # Assuming probabilities are scalars

        return images, probability
    



# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts images to PyTorch tensors
    transforms.Resize((300, 300)),
])



# Create datasets
train_dataset = CustomDataset(train_set, train_proba, transform=transform)
test_dataset = CustomDataset(test_set, test_proba, transform=transform)
val_dataset = CustomDataset(val_set, val_proba,transform=transform) 
# print(type(train_dataset[0]))
# print(type(test_dataset[0]))
# print(type(val_dataset[0]))

print(type(train_dataset[0][0]))
print(type(train_dataset[0][1]))

print(train_dataset[0][0].shape)
print(train_dataset[0][1].shape)

print(train_dataset[0][0])
print(train_dataset[0][1])


# Create data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


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
    
# Lists to store training loss and accuracy
training_loss_history = []
training_accuracy_history = []
model = OneChannel_CNN_Probability().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_function = nn.BCELoss()  # Binary Cross-Entropy Loss
# Define the number of training epochs
num_epochs = 41

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for image, probability in train_loader:
        image = image.to(device)
        probability = probability.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        
        # Calculate the loss
        loss = loss_function(outputs, probability.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate training accuracy
        predicted_labels = (outputs > 0.5).float()  # Convert probabilities to binary labels
        correct_predictions += (predicted_labels == probability).sum().item()
        total_samples += image.size(0)

    avg_loss = total_loss / len(train_loader)
    training_accuracy = correct_predictions / total_samples

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {training_accuracy:.4f}")
    training_loss_history.append(avg_loss)
    training_accuracy_history.append(training_accuracy)



    


training_accuracy_history = np.array(training_accuracy_history)*100 # Convert to numpy array and convert to percentage

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot training loss on the first subplot
ax1.plot(training_loss_history, label="Training Loss")
ax1.set_yscale('log')  # Set the y-axis to a log scale for loss
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss (Log Scale)")
ax1.legend()

# Plot training accuracy on the second subplot
ax2.plot(training_accuracy_history, label="Training Accuracy", color='orange')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training Accuracy")
ax2.legend()

plt.tight_layout()  # Adjust subplots to prevent overlap
plt.show()



# save model 
torch.save(model.state_dict(), r'F:\school work\Fall_2023_USC\AME_505_Engineering_Information_Modeling\AME505_final_project\elpv-dataset-1.0\elpv-dataset-1.0\utils\OneChannel_CNN_Probability.pth')

#validate the model
#load the model
model = OneChannel_CNN_Probability().to(device)
model.load_state_dict(torch.load(r'F:\school work\Fall_2023_USC\AME_505_Engineering_Information_Modeling\AME505_final_project\elpv-dataset-1.0\elpv-dataset-1.0\utils\OneChannel_CNN_Probability.pth'))

def validate_model(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for image, probability in dataloader:
            image = image.to(device)
            probability = probability.to(device)
            outputs = model(image)
            predicted_labels = (outputs > 0.5).float()
            correct_predictions += (predicted_labels == probability).sum().item()
            total_samples += image.size(0)

    validation_accuracy = correct_predictions / total_samples
    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    return validation_accuracy

validate_model(model, val_loader, device)



#test the model

# Lists to store test loss and accuracy
test_loss_history = []
test_accuracy_history = []

# Test loop
total_loss = 0.0
correct_predictions = 0
total_samples = 0

for image, probability in test_loader:
    image = image.to(device)
    probability = probability.to(device)
    outputs = model(image)

    # Calculate the loss
    loss = loss_function(outputs, probability.float())
    total_loss += loss.item()

    # Calculate test accuracy
    predicted_labels = (outputs > 0.5).float()  # Convert probabilities to binary labels
    correct_predictions += (predicted_labels == probability).sum().item()
    total_samples += image.size(0)

avg_loss = total_loss / len(test_loader)
test_accuracy = correct_predictions / total_samples

print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
test_loss_history.append(avg_loss)
test_accuracy_history.append(test_accuracy)

test_accuracy_history = np.array(test_accuracy_history)*100  # Convert to numpy array and convert to percentage



