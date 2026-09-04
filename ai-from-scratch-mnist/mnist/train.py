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


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

lin = torch.nn.Linear(784, 10).to(device)
optimizer = torch.optim.SGD(lin.parameters(), lr=0.01)
loss = torch.nn.CrossEntropyLoss()

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
      print(f"Batch {batch_num} | Loss: {loss_value.item()} | Accuracy: {accuracy}%")   