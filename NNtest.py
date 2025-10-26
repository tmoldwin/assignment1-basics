import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as dl
from torchvision.transforms import ToTensor
from sklearn.metrics import ConfusionMatrixDisplay


fashion_mnist = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True
)
print(fashion_mnist.data.shape)
print(fashion_mnist.targets.shape)
# plt.imshow(fashion_mnist.data[0], cmap='gray')
# plt.show()
img_size = fashion_mnist.data[0].shape
num_classes = len(torch.unique(fashion_mnist.targets))
print(img_size)
input_size = fashion_mnist.data[0].numel()


class my_model(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 68, kernel_size = 3, padding = 1),
            nn.ReLU(),

            nn.Conv2d(68, 68, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),

            nn.Linear(int(img_size[0]*(1/2**2)*img_size[1]*(1/2**2) * 68), 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.SELU(),
            nn.Linear(200, num_classes),
        )

    def forward(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    model = my_model(input_size, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    train_size = 60000  
    # Subsample size
    full_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=ToTensor()
    )
    indices = torch.randperm(len(full_dataset))[:train_size]
    train_dataset = torch.utils.data.Subset(full_dataset, indices)
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=ToTensor()
    )
    print(train_dataset[0])
    train_loader = dl.DataLoader(train_dataset, batch_size=30)
    full_train_load = dl.DataLoader(train_dataset, batch_size = len(train_dataset))
    full_test_load = dl.DataLoader(test_dataset, batch_size = len(test_dataset))
    losses_train = [];
    losses_test = [];
    for epoch in range(10):
        print("epoch", {epoch})
        for idx, batch in enumerate(train_loader):
            x, y = batch
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {idx}, Loss: {loss.item()}")
        for idx, batch in enumerate(full_train_load):
            x_train_full, y_train_full = batch
            y_hat_train_full = model(x_train_full)
            train_loss = loss_fn(y_hat_train_full, y_train_full)
            losses_train.append(train_loss.item())
            print(f"Full Train Epoch: {epoch}, Batch: {idx}, Losses: {losses_train}")
        for idx, batch in enumerate(full_test_load):
            x_test_full, y_test_full = batch
            y_hat_test_full = model(x_test_full)
            test_loss = loss_fn(y_hat_test_full, y_test_full)
            losses_test.append(test_loss.item())
            print(f"Full Test Epoch: {epoch}, Batch: {idx}, Losses: {losses_test}")

fig1, axes = plt.subplots(1,3)
axes[0].plot(losses_train)
axes[0].plot(losses_test)
plt.show()



        


