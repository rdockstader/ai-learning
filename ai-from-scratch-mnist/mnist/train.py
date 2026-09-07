import torch
import torchvision
import csv
import datetime as dt
from torch.utils.data import DataLoader
from pathlib import Path

#constants
checkpoints_folder_name = "checkpoints"
csv_name = "results.csv"

Path(checkpoints_folder_name).mkdir(exist_ok=True)

# tuning variables
save_checkpoint = False
use_manual_seed = True
manual_seed = 42
epoch_count = 5
batch_size = 64
learning_rate = 0.01
model = "linear"

# other variables
checkpoint_file_name = f"mnist_{model}"
if use_manual_seed:
   checkpoint_file_name += f"_ms{manual_seed}"

checkpoint_file_name += f"_b{batch_size}_e{epoch_count}_lr{learning_rate}"

checkpoint_file_name += ".pt"


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tensor_transform = torchvision.transforms.Compose([
   torchvision.transforms.ToTensor()
])

if use_manual_seed:
   torch.manual_seed(manual_seed)
   print(f"Using manual seed {manual_seed}")

train_dataset = torchvision.datasets.MNIST("./data", transform=tensor_transform, download=True)
test_dataset = torchvision.datasets.MNIST("./data", transform=tensor_transform, train=False, download=True)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


lin = torch.nn.Linear(784, 10).to(device)
optimizer = torch.optim.SGD(lin.parameters(), lr=learning_rate)
loss = torch.nn.CrossEntropyLoss()


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
test_accuracy = round(test_accuracy, 2)

print(f"accuracy after {epoch_count} epochs: {test_accuracy}%")

if save_checkpoint: 
   torch.save(lin.state_dict(), f"{checkpoints_folder_name}/{checkpoint_file_name}")




file_exists = Path(csv_name).exists()

with open(csv_name, "a", newline="") as file:
   writer = csv.writer(file)

   if not file_exists:      
      writer.writerow(["timestamp", "model", "seed", "batch_size", "epochs", 
                       "learning_rate", "test_accuracy"])    

   if use_manual_seed:
      seed = manual_seed
   else:
      seed = "N/A"

   writer.writerow([dt.datetime.now().isoformat(), model, seed, batch_size, epoch_count,
                    learning_rate, test_accuracy])