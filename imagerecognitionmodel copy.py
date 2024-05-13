import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define the CNN architecture

def main(num_epochs, batch_size, num_fc_layers, lr, fc_layer_size, kernel_size): 
    # Define the CNN architecture
    class CNN(nn.Module):
        def __init__(self, num_fc_layers, kernel_size, fc_layer_size):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_size, padding='same')
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
            self.pool = nn.MaxPool2d(2, 2)
            self.fc_layers = nn.ModuleList()
            self.fc_layers.append(nn.Linear(128 * 4 * 4, fc_layer_size))
            for i in range(num_fc_layers):
                self.fc_layers.append(nn.Linear(fc_layer_size, fc_layer_size) if i < num_fc_layers else nn.Linear(fc_layer_size, 10))
            self.fc_layers.append(nn.Linear(fc_layer_size, 10))

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = x.view(-1, 128 * 4 * 4)
            for i, fc_layer in enumerate(self.fc_layers):
                x = fc_layer(x)
                if i < len(self.fc_layers) - 1:
                    x = torch.relu(x)
            return x
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Initialize the network, loss function, and optimizer
    net = CNN(num_fc_layers, kernel_size, fc_layer_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Train the network
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # Test the network on the test data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    return 100 * correct / total