import os

import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import shap


class MNISTNET(nn.Module):
    def __init__(self):
        super(MNISTNET, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(1600, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.view(128, 1600)
        x = self.fc_layers(x)
        return x


batch_size = 128
num_epochs = 5
device = torch.device('cuda:0')


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.log(), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output.log(), target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


# Getting the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_set = datasets.MNIST('data/', train=True, download=True, transform=transform)
test_set = datasets.MNIST('data/', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

# Training
model = MNISTNET().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

model_name = "mnist_model"

if not os.path.exists(model_name):

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    torch.save(model.state_dict(), model_name)
else:
    model = MNISTNET()
    model.load_state_dict(torch.load(model_name))
    model = model.to(device)
    model.eval()
    test(model, device, test_loader)



# shap values
# batch = next(iter(test_loader))
# images, _ = batch
#
# background = images[:100].to(device)
# test_images = images[100:105].to(device)
#
# e = shap.DeepExplainer(model, background)
# shap_values = e.shap_values(test_images)
#
# shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
# test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)
# shap.image_plot(shap_numpy, -test_numpy)