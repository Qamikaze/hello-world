"""
This code evaluates the Image recognition performance in terms of LOSS
brief code structure:
    - Data Loading / Preparation (MNIST)
    - Defining CNN
    - Train data (with Validation)
    - Test set
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator # For integer valued x-axis for Epochs in plot
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler

import torch.optim as optim

from datetime import datetime

# Execution time
startTime = datetime.now()

# Plot specifications
plt.style.use('bmh')
# Epoch x-axis, integers only
ax = plt.figure().gca() 
ax.xaxis.set_major_locator(MaxNLocator(integer=True))






"""
Hyperparameters / additional info specification
"""
# Hyperparameters
Learn_Rate = 0.001
Batch_Size = 4
# Proportion training data used for validation
Validation_Split = 0.2
# Amount of training epochs
Epochs = 3
    


"""  
Loading and normalizing the data
"""
# Transformation specification for traning/test data
transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])

def train_data_split(validation_split):
    
    np.random.seed(123)
    
    train_indices = 100
    val_indices= 10
    train_loader = zip(np.random.randn(train_indices, 100), np.random.randn(train_indices, 1))
    validation_loader = zip(np.random.randn(val_indices, 100), np.random.randn(val_indices, 1))

    return train_loader, validation_loader, train_indices, val_indices

# Calling the train_data_split function 
train_loader, validation_loader, train_size, validation_size = train_data_split(Validation_Split)



# Defining the NN3
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100 , 32)          # INSERT DIMENSION INPUT
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x    

net = Net()



# Count number of learnable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"learnable parameters: {count_parameters(Net())}\n")



# Defining the Optimizer and Loss function (L1 loss)
l1_lambda = 0.001
criterion = torch.nn.MSELoss()            
optimizer = optim.Adam(net.parameters(), lr= Learn_Rate)



# Train Data 
print("Start Training")
plot_list_T = []
plot_list_V = []
for epoch in range(Epochs):  # loop over the dataset multiple times
    print(f"\nEpoch {epoch+1}/{Epochs}:")
    
    # initialization
    running_loss = 0.0
    running_loss_total = 0.0
 
    
    # Training 
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
   
        optimizer.zero_grad()
        outputs = net(inputs)

        l1_penalty = sum(p.abs().sum() for p in net.parameters())
        loss = criterion(outputs, labels) +  l1_penalty
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_total += loss.item()

            
    # print/save loss (Training)
    plot_list_T.append(running_loss_total * Batch_Size / train_size)
    print(f"loss T = {running_loss_total * Batch_Size / train_size}")   
            
    # Validation 
    running_loss = 0.0
    for j, data in enumerate(validation_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        
        # forward
        outputs = net(inputs)
        l1_penalty = sum(p.abs().sum() for p in net.parameters())
        loss = criterion(outputs, labels) +  l1_penalty
        
        running_loss += loss.item()
    # print/save loss (Training)
    plot_list_V.append(running_loss * Batch_Size / validation_size)
    print(f"loss V = {running_loss * Batch_Size / validation_size}\n")

# Check
print(f"\nplot_list_T \n {plot_list_T}")
print(f"\nplot_list_V \n {plot_list_V}")

# Plot
plt.plot(range(1, Epochs + 1), plot_list_T, label = f"{Learn_Rate}")
plt.plot(range(1, Epochs + 1), plot_list_V, label = "Validation")


plt.style.use('bmh')

plt.xlabel("epoch")  
plt.ylabel("loss")
plt.title("Comparing Training / Validation (loss)")
plt.legend(loc="upper right")
plt.show()

print("\nFinished Training\n")


      
# Print exectution time
print(f"\ntotal runtime: {datetime.now() - startTime}")
