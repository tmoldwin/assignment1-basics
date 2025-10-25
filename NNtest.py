import torch
import torchvision
import matplotlib.pyplot as plt

fashion_mnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
print(fashion_mnist.data.shape)
print(fashion_mnist.targets.shape)
plt.imshow(fashion_mnist.data[0], cmap='gray')
plt.show()

data_size = fashion_mnist.data[0].shape
num_classes = len(torch.unique(fashion_mnist.targets))
print(data_size)

class my_model(torch.nn.Module):
    def init(self, input_size, num_classes):

    def forward(self, x):
        pass;







