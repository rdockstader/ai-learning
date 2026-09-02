# import torch libraries for AI stuffs
import torch
import torchvision
from torch.utils.data import DataLoader


# print the version of torch we have
# print(torch.__version__)

# store if we have cuda available
cuda_is_available = torch.cuda.is_available()

# print(f"CUDA Available: {cuda_is_available}")

# if cuda is available, print hte device and set the torch device to cuda
if cuda_is_available:
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# print the device we are using
print(device)

# Download the MNIST dataset, and then grab an image from it
tensor_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

train_dataset = torchvision.datasets.MNIST("./data", transform=tensor_transform, download=True)

# print(len(train_dataset))

image, label = train_dataset[0]

# print(image)
# print(label)

# print(type(image))
# print(image.shape)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

lin = torch.nn.Linear(784, 10).to(device)
optimizer = torch.optim.SGD(lin.parameters(), lr=0.01)
loss = torch.nn.CrossEntropyLoss()

## run the training in a loop
for batch_num, (images, labels) in enumerate(train_loader):
    # load up the GPU
    images = images.to(device)
    labels = labels.to(device)

    # clear old gradients
    optimizer.zero_grad()

    # print(images.shape)
    # print(labels.shape)

    # flatten images
    flattened_imgs = torch.flatten(images, start_dim=1)
    # print(flattened_imgs.shape)

    # forward pass
    lin_output = lin(flattened_imgs)
    # print(lin_output.shape)
    # print(lin_output[0])
    # print(labels[0])

    # check accuracy
    predictions = lin_output.argmax(dim=1)


    accuracy = (predictions == labels).float().mean().item() * 100
    # print(accuracy)
    # calculate loss
    # prediction = lin_output[0].argmax()
    # print(f"Actual: {labels[0]} | Predicted: {prediction}")

    loss_value = loss(lin_output, labels)
    # print(loss_value.item())

    # backward pass
    loss_value.backward()

    # optimizer step
    # print(f"Before optimizer: {lin.weight[0][400].item()}")
    optimizer.step()
    # print(f"After optimizer: {lin.weight[0][400].item()}")

    # print(lin.weight.shape)
    # print(lin.weight.grad.shape)
    # print(lin.weight.grad.abs().max())

    if batch_num % 100 == 0:
        print(f"Batch {batch_num} | Loss: {loss_value.item()} | Accuracy: {accuracy}%")










