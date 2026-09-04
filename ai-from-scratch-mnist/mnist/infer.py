import random
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tensor_transform = torchvision.transforms.Compose([
   torchvision.transforms.ToTensor()
])


test_dataset = torchvision.datasets.MNIST("./data", transform=tensor_transform, train=False, download=True)

lin = torch.nn.Linear(784, 10).to(device)
lin.load_state_dict(torch.load("checkpoints/mnist_linear.pt"))


# test inference
image, label = random.choice(test_dataset)

orig_img = torch.squeeze(image)

image = image.to(device)

image = torch.unsqueeze(image, 0)
image = image.flatten(start_dim=1)

output = lin(image)

prediction = output.argmax(dim=1).item()

plt.imshow(orig_img, cmap="gray")
plt.title(f"Inference result:: Prediction: {prediction}, actual: {label}")
print(f"Inference result:: Prediction: {prediction}, actual: {label}")
plt.show()