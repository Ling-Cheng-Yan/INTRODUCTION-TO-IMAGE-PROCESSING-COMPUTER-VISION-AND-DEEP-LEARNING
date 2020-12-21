import os
import cv2
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import csv


DIR_TRAIN = "train/train/"
DIR_TEST = "test1/test1/"

### Checking Data Format
imgs = os.listdir(DIR_TRAIN) 
test_imgs = os.listdir(DIR_TEST)

### Class Distribution
dogs_list = [img for img in imgs if img.split(".")[0] == "dog"]
cats_list = [img for img in imgs if img.split(".")[0] == "cat"]


class_to_int = {"dog" : 0, "cat" : 1}
int_to_class = {0 : "dog", 1 : "cat"}

### Transforms for image - ToTensor and other augmentations
def get_train_transform():
    return T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0, 0, 0),(1, 1, 1))
    ])
    
def get_val_transform():
    return T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0, 0, 0),(1, 1, 1))
    ])

### Dataset Class - for retriving images and labels
class CatDogDataset(Dataset):
    
    def __init__(self, imgs, class_to_int, mode = "train", transforms = None):
        
        super().__init__()
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms
        
    def __getitem__(self, idx):
        
        image_name = self.imgs[idx]
        
        ### Reading, converting and normalizing image
        #img = cv2.imread(DIR_TRAIN + image_name, cv2.IMREAD_COLOR)
        #img = cv2.resize(img, (224,224))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        #img /= 255.
        img = Image.open(DIR_TRAIN + image_name)
        img = img.resize((224, 224))
        
        if self.mode == "train" or self.mode == "val":
        
            ### Preparing class label
            label = self.class_to_int[image_name.split(".")[0]]
            label = torch.tensor(label, dtype = torch.float32)

            ### Apply Transforms on image
            img = self.transforms(img)

            return img, label
        
        elif self.mode == "test":
            
            ### Apply Transforms on image
            img = self.transforms(img)

            return img
            
        
    def __len__(self):
        return len(self.imgs)

### Splitting data into train and val sets
train_imgs, val_imgs = train_test_split(imgs, test_size = 0.25)

### Dataloaders
train_dataset = CatDogDataset(train_imgs, class_to_int, mode = "train", transforms = get_train_transform())
val_dataset = CatDogDataset(val_imgs, class_to_int, mode = "val", transforms = get_val_transform())
test_dataset = CatDogDataset(test_imgs, class_to_int, mode = "test", transforms = get_val_transform())

train_data_loader = DataLoader(
    dataset = train_dataset,
    num_workers = 4,
    batch_size = 16,
    shuffle = True
)

val_data_loader = DataLoader(
    dataset = val_dataset,
    num_workers = 4,
    batch_size = 16,
    shuffle = True
)

test_data_loader = DataLoader(
    dataset = test_dataset,
    num_workers = 4,
    batch_size = 16,
    shuffle = True
)

### GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

### Function to calculate accuracy
def accuracy(preds, trues):
    
    ### Converting preds to 0 or 1
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    
    ### Calculating accuracy by comparing predictions with true labels
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    
    ### Summing over all correct predictions
    acc = np.sum(acc) / len(preds)
    
    return (acc * 100)

### Function - One Epoch Train
def train_one_epoch(train_data_loader):
    
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    ###Iterating over data loader
    for images, labels in train_data_loader:
        
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
        #Reseting Gradients
        optimizer.zero_grad()
        
        #Forward
        preds = model(images)
        
        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
        
        #Backward
        _loss.backward()
        optimizer.step()
    
    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    
    ###Storing results to logs
    train_logs["loss"].append(epoch_loss)
    train_logs["accuracy"].append(epoch_acc)
    train_logs["time"].append(total_time)
        
    return epoch_loss, epoch_acc, total_time

### Function - One Epoch Valid
def val_one_epoch(val_data_loader, best_val_acc):
    
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    ###Iterating over data loader
    for images, labels in val_data_loader:
        
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
        #Forward
        preds = model(images)
        
        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
    
    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    
    ###Storing results to logs
    val_logs["loss"].append(epoch_loss)
    val_logs["accuracy"].append(epoch_acc)
    val_logs["time"].append(total_time)
    
    ###Saving best model
    if epoch_acc > best_val_acc:
        best_val_acc = epoch_acc
        torch.save(model.state_dict(),"resnet50(Resize).pth")
        
    return epoch_loss, epoch_acc, total_time, best_val_acc


### ResNet50 Pretrained Model
model = resnet50(pretrained = True)

# Modifying Head - classifier

model.fc = nn.Sequential(
    nn.Linear(2048, 1, bias = True),
    nn.Sigmoid()
)

### Defining model parameters

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
learning_rate = 0.0001
batch_size = 32
# Learning Rate Scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)

#Loss Function
criterion = nn.BCELoss()

# Logs - Helpful for plotting after training finishes
train_logs = {"loss" : [], "accuracy" : [], "time" : []}
val_logs = {"loss" : [], "accuracy" : [], "time" : []}

# Loading model to device
model.to(device)

# The number of epochs 
epochs = 5

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def main():
    torch.multiprocessing.freeze_support()
    print('waiting...')
    swriter = SummaryWriter()
    # Get a batch of training data
    inputs, classes = next(iter(train_data_loader))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_to_int for x in classes])
    plt.show()

    with open('Q5(Resize).csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        ### Training and Validation
        best_val_acc = 0
        for epoch in range(epochs):
            ###Training
            loss, acc, _time = train_one_epoch(train_data_loader)
            swriter.add_scalar("Loss/train", loss, epoch)
            swriter.add_scalar("Accuracy/train", acc, epoch)
            #Print Epoch Details
            print("\nTraining")
            print("Epoch {}".format(epoch+1))
            print("Loss : {}".format(round(loss, 4)))
            print("Acc : {}".format(round(acc, 4)))
            print("Time : {}".format(round(_time, 4)))
            # csv
            writer.writerow(['Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f}'.format(
            epoch, loss, acc)])
            
            ###Validation
            loss, acc, _time, best_val_acc = val_one_epoch(val_data_loader, best_val_acc)
            
            #Print Epoch Details
            print("\nValidating")
            print("Epoch {}".format(epoch+1))
            print("Loss : {}".format(round(loss, 4)))
            print("Acc : {}".format(round(acc, 4)))
            print("Time : {}".format(round(_time, 4)))
            # csv
            writer.writerow(['Epoch: {} \tValidating Loss: {:.6f} \tValidating Accuracy: {:.6f}'.format(
            epoch, loss, acc)])
        # csv
        writer.writerow(['learning_rate: {} '.format(float(learning_rate))])
        writer.writerow(['batch_size: {} '.format(int(batch_size))])
        writer.writerow(['num_epochs: {} '.format(int(epochs))])
    # print model
    summary(model, (3, 24, 40))

    ### Plotting Results

    #Loss
    plt.title("Loss")
    plt.plot(np.arange(1, 6, 1), train_logs["loss"], color = 'blue')
    plt.plot(np.arange(1, 6, 1), val_logs["loss"], color = 'yellow')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    #Accuracy
    plt.title("Accuracy")
    plt.plot(np.arange(1, 6, 1), train_logs["accuracy"], color = 'blue')
    plt.plot(np.arange(1, 6, 1), val_logs["accuracy"], color = 'yellow')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
    swriter.flush()


if __name__ == '__main__':
    main()