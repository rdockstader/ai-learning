import torch
import torchvision
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tensor_transform = torchvision.transforms.Compose([
   torchvision.transforms.ToTensor()
])

train_dataset = torchvision.datasets.MNIST("./data", transform=tensor_transform, download=True)
test_dataset = torchvision.datasets.MNIST("./data", transform=tensor_transform, train=False, download=True)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(len(train_dataset))
print(len(test_dataset))

lin = torch.nn.Linear(784, 10).to(device)
optimizer = torch.optim.SGD(lin.parameters(), lr=0.01)
loss = torch.nn.CrossEntropyLoss()

epoch_count = 5

for epoch_num in range(epoch_count):
   for batch_num, (images, labels) in enumerate(train_loader):

      images = images.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      images = images.flatten(start_dim=1)

      outputs = lin(images)

      predictions = outputs.argmax(dim=1)
      accuracy = (predictions == labels).float().mean().item() * 100

      loss_value = loss(outputs, labels)
      loss_value.backward()

      optimizer.step()

      if batch_num % 100 == 0:
         print(f"Epoch {epoch_num + 1} Batch {batch_num} | Loss: {loss_value.item()} | Accuracy: {accuracy}%")   


# Test Loop ()
total = 0
correct = 0
with torch.no_grad():
   for batch_num, (images, labels) in enumerate(test_loader):
      images = images.to(device)
      labels = labels.to(device)

      images = images.flatten(start_dim=1)

      outputs = lin(images)

      predictions = outputs.argmax(dim=1)
      correct += (predictions == labels).sum().item()
      total += len(images)


test_accuracy = correct / total * 100

print(f"accuracy after {epoch_count} epochs: {test_accuracy}%")

torch.save(lin.state_dict(), "checkpoints/mnist_linear.pt")