import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from torchsummary import summary

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Cifar-10 data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)
testLoader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)

# Data classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device)

# Parameters
criterion = nn.CrossEntropyLoss()
lr = 0.00116
epochs = 3
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)


'''
Q5.4
'''
print("--------Q5.4's Answer--------")
print("Training your model at least 20 epochs by your own computer, save your model and take a screenshot of your training loss and accuracy. No saved images no points.")
# Train
print("------Training------")
training_losses = []
for epoch in range(epochs):
    training_loss = 0.0

    for times, data in enumerate(trainLoader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        training_loss += (loss.item()/2000)
        training_losses.append(training_loss)

        if times % 100 == 99 or times+1 == len(trainLoader):
            print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, times+1, len(trainLoader), training_loss))

print('Finished Training')

# Test
corrects = []
correct = 0
total = 0
with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        corrects.append(100*correct/total)

print('Accuracy of the network on the 10000 test inputs: %d %%' % (100*correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(8):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

plt.plot(training_losses, label='Training loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.show()

plt.plot(corrects, label='Testing accuracy')
plt.xlabel("Epochs")
plt.ylabel("%")
plt.legend(frameon=False)
plt.show()




'''
Q5.2
'''
print("--------Q5.2's Answer--------")
print("Print out training hyperparameters (batch size, learning rate, optimizer). ")
print("batch size: 8")
print("learning rate:", lr)
print("optimizer:", optimizer)

'''
Q5.3
'''
print("--------Q5.3's Answer--------")
print("Construct and show your model structure. (You can use available architecture provided by ML framework to build your model)")
summary(net,(3, 32, 32))


'''
Q5.5
'''
print("--------Q5.5's Answer--------")
print("Load your model trained at 5.4, let us choose one image from test images, inference the image, show image and estimate the image as following. No saved model no points.")
print("------Testing Image Index------")
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

# Save the model
torch.save(net.state_dict(), 'cifar10-model.ckpt')


