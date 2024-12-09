import torch
import os
import time
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
# Define the indices for the samples you want to include in the subset
# For example, let's sample the first 1000 samples
# sample_indices = list(range(30000))

# # Create a subset of the dataset
# training_set = Subset(training_set, sample_indices)

print('Training set has {} instances'.format(len(training_set)))

train_loader = torch.utils.data.DataLoader(training_set, batch_size=1000, shuffle=True, num_workers=os.cpu_count(),
                                           pin_memory = True, persistent_workers = True)

# PyTorch models inherit from torch.nn.Module
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 240)
        self.fc3 = nn.Linear(240, 480)
        self.fc4 = nn.Linear(480, 240)
        self.fc5 = nn.Linear(240, 120)
        self.fc6 = nn.Linear(120, 100)
        self.fc7 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))

        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x
    
# Create an instance of the model
model = ImageClassifier().cuda()

# We will be using CrossEntropyLoss as our loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Create CUDA streams  
stream1 = torch.cuda.Stream()  
stream2 = torch.cuda.Stream()  

# Initialize GradScaler for AMP
scaler = GradScaler()

start = time.time()
# Start profiling
torch.cuda.profiler.start()
# Training loop
for epoch in range(3):

    torch.cuda.nvtx.range_push(f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx%10 == 0:
            data, target = data.cuda(non_blocking = True), target.cuda(non_blocking = True)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output = model(data)
                loss = loss_fn(output, target)
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Move to the next batch
            # data, target = next_data, next_target
            
    print(f'Epoch {epoch}, Loss: {loss.item()}')
    torch.cuda.nvtx.range_pop()

# Stop profiling
torch.cuda.profiler.stop()
end = time.time()
print("Time taken: ", end-start)

train_loader._iterator._shutdown_workers()
