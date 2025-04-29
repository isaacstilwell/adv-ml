import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


device =  "cuda" if torch.cuda.is_available() else "cpu"


'''
Function description:
Function load_data takes a string data_folder as input
Input data_folder is path to the FashionMNIST dataset
We explain how to load dataset for server users and local machine users in main
See under: if __name__ == '__main__'
Choose the one that applies to you
Function data_folder returns two datasets (training_set and validation_set)
'''
def load_dataset(data_folder):
    # TODO:
    # Define transformations of the dataset
    # Transformations can be found in https://pytorch.org/vision/0.9/transforms.html
    # For training data:
    #    First convert the data to tensor
    #    Then normalize the data with mean 0.5 and std 0.5
    #    Choose to include data augmentation by adding transformation(s)
    #    You can use RandomHorizontalFlip and/or RandomVerticalFlip
    #    We will evaluate on augmented dataset using RandomHorizontalFlip and RandomVerticalFlip
    # For validation data:
    #    First convert the data to tensor
    #    Then normalize the data with mean 0.5 and std 0.5
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Load the dataset from data_folder
    # Use different transformations to augment training and validation dataset
    training_set = torchvision.datasets.FashionMNIST(data_folder, train=True, transform=train_transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST(data_folder, train=False, transform=val_transform, download=True)

    return training_set, validation_set



'''
Function description:
The class FashionCNN defines the model architecture of the CNN model we use to classifiy FashionMNIST
'''
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()

        # TODO:
        # Fill in the missing parameters of Conv2d:
        #   (1) The data has 1 in_channel, 6 out_channels
        #   (2) The convolution has kernal size 5 and padding 0
        # Fill in the missing parameters of MaxPool2d
        #   (1) The max pool has kernal size 2
        #   (2) The max pool has stride 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # TODO:
        # Define a second layer of the CNN in a similar way of layer1:
        # The layer has sequential connection of the following functions:
        #   (1) A 2d convolution with 16 out_channels, kernel size 5 and padding 0
        #   (2) A ReLU activation function
        #   (3) A max pool with kernel size 2 and stride 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)

        # TODO:
        # Fill in the missing parameters of Linear in the second fully connected layer
        # The fully connected layer has 84 output features
        self.fc2 = nn.Linear(in_features=120, out_features=84)

        # TODO:
        # Define a third fully connected layer with 10 outputs
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    # The forward defines how the model is excuted when data x comes in
    def forward(self, x):
        out = self.layer1(x)

        # TODO:
        # Fill in the output that runs through second convolutional layer
        out = self.layer2(out)

        out = out.view(-1, 16 * 4 * 4)
        out = F.relu(self.fc1(out))

        # TODO:
        # Fill in the output that runs through second fully connected layer with ReLU activation function
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out



'''
Function description:
Function evaluate_model takes a model and a data loader as inputs
Input model is the model to be evaluated
Input val_loader is a data loader for the validation dataset
Function evaluate_model returns the accuracy of the model on the given validation data
'''
def evaluate_model(model, val_loader):
    # Set the model to evaluation mode
    model.eval()
    model.to(device)

    total = torch.tensor(0, device=device)
    correct = torch.tensor(0, device=device)
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            # TODO:
            # Complete the eveluation process
            #   Get outputs from the model on the current batch of images
            #   Get the predicted labels of the current batch of images from the model outputs
            #   Compare the predicted labels with the labels to find which ones are correct
            #   Also increase the count for the total number of data used in validation
            outputs = model.forward(images)
            _, predictions = torch.max(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)

    # TODO:
    # Complete the computation for accuracy: 0 <= accuracy <= 1
    accuracy = correct / total

    assert((accuracy >= 0) and (accuracy <= 1))
    return accuracy.detach().cpu().item()


'''
Function description:
Function evaluate_model_from_path takes a path to a trained FashionCNN model and a data loader as inputs
Input model_path for example is './model.p'
Input val_loader is a data loader for the validation dataset
Function evaluate_model_from_path loads the model from the path and evaluates the model using function evaluate_model
This function is provided so that you can evlaute a model that is already trained
'''
def evaluate_model_from_path(model_path, val_loader):
    model = FashionCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return evaluate_model(model, val_loader)


'''
Function description:
A plotting function to visualize the change of loss and accuracy with training epochs
The plot is saved in the current working directory as a jpg file
'''
def plot_lossacc(loss_list, accuracy_list):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    epochs = len(loss_list)
    epochs_list = list(range(1, epochs+1))
    ax1.set_ylim(min(loss_list), max(loss_list))
    ax1.set_ylabel("Loss", fontsize=20)
    line1 = ax1.plot(epochs_list, loss_list, color='red', lw=4, label="Loss")
    ax1.set_xlabel("Epochs", fontsize=20)
    ax1.set_xticks(epochs_list, labels=[str(ep) for ep in epochs_list], fontsize=15)
    ax1.tick_params(axis='y', which='major', labelsize=15)

    ax2 = ax1.twinx()
    ax2.set_ylim(math.floor(min(accuracy_list)*100)/100, 1)
    ax2.set_ylabel("Accuracy", fontsize=20)
    line2 = ax2.plot(epochs_list, accuracy_list, color='green', lw=4, label="Accuracy")
    ax2.tick_params(axis='y', which='major', labelsize=15)

    ax1.set_title('Loss and Accuracy of {} Epochs Training'.format(epochs), fontsize=20)
    lines = line1+line2
    labs = [line.get_label() for line in lines]
    ax1.legend(lines, labs, loc=0, fontsize=15)

    # The plot is saved with the name for example loss_accuracy_2epochs.jpg
    # You may modify this to save the plots with different names
    plt.savefig(f"./loss_accuracy_{epochs}epochs.jpg")



'''
Function description:
Function train_model takes the training dataset and the validation dataset as inputs
The function loads the datasets into dataloaders and then train a CNN model on the training dataset
At the end of each training epoch, the current model is evaluated on the validation dataset
Training loss and validation accuracy is printed for each epoch
When training is finished, the trained model is saved in the current working directory
A plot on how training loss and validation accuracy change with the epochs will be saved in a jpg file
'''
def train_model(training_set, validation_set):

    model = FashionCNN()
    model.to(device)

    # TODO:
    # Define the learning rate, optimizer and training batch size
    # You may try different values for these variables and find the ones that work well
    # Optimizer can be found at https://pytorch.org/docs/stable/optim.html
    # You can choose SGD or Adam optimizer
    learning_rate = 0.000125
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = 4

    # Load the training and validation data to data loaders
    # Training data will be shuffled but validation data does not need to be shuffled
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    # TODO:
    # Define the number of training epochs
    # You may try different values of training epochs and find the one that works well
    # num_epochs should be no larger than 50
    num_epochs = 25

    assert(num_epochs <= 50)

    # Record the training loss and validation accuracy of each epoch
    loss_list = []
    accuracy_list = []

    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()
        # Record the training loss of each batch
        epoch_loss = []
        # Training data is shuffled for each epoch
        # Shuffling happens when the iteration is initialized at the beginning of each epoch
        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            # TODO:
            # Complete the training process:
            #   Get outputs from the model on the current batch of images
            #   Compute the cross-entropy loss given the current outputs and labels
            #   Initialize the gradient as 0 for the current batch
            #   Backpropagate the loss
            #   Update the model parameters
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

        accuracy = evaluate_model(model, val_loader)

        loss_list.append(np.mean(epoch_loss))
        accuracy_list.append(accuracy)
        print(f"Epoch: {epoch}, Training Loss: {loss_list[-1]:.2f}, Validation Accuracy: {accuracy*100:.2f}%")

    plot_lossacc(loss_list, accuracy_list)


    # The trained model is saved in the current working directory
    # You can save the model using different names
    # BUT only submit model.p to Gradescope
    # Our grading system do not accept models with other names
    torch.save(model.state_dict(), "./model.p")



if __name__ == '__main__':
    # TODO:
    # If you are working on the server floo.cs.uchicago.edu, you should use the line below
    # training_set, validation_set = load_dataset('/var/cs25800/hw1/data/FashionMNIST')
    #
    # If you are working on your own laptop, you should use the line below
    # This will create a folder named FashionMNIST in your current working directory
    training_set, validation_set = load_dataset('./FashionMNIST')

    train_model(training_set, validation_set)

