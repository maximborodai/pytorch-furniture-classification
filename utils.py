import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image, ImageOps
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# function to resize image
def resize_image(src_image, size=(128, 128), bg_color="white"):
    # resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.Resampling.LANCZOS)

    # Create a new square background image
    new_image = Image.new("RGB", size, bg_color)

    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))

    # return the resized image
    return new_image


#def load_dataset(train_folder_path, test_folder_path):
def load_dataset(data_path):
    # Load all the images
    transformation = transforms.Compose(
        [
            # Randomly augment the image data
            # Random horizontal flip
            transforms.RandomHorizontalFlip(0.5),
            # Random vertical flip
            transforms.RandomVerticalFlip(0.3),
            # transform to tensors
            transforms.ToTensor(),
            # Normalize the pixel values (in R, G, and B channels)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # Load all of the images, transforming them
    full_dataset = torchvision.datasets.ImageFolder(
        #root=train_folder_path,
        root=data_path,
        transform=transformation
    )
    '''
    test_dataset = torchvision.datasets.ImageFolder(
        #root=test_folder_path,
        transform=transformation
    )
    '''

    ''''''
    # Split into training (70% and testing (30%) datasets)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # use torch.utils.data.random_split for training/test split
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # define a loader for the training data we can iterate through in 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )

    # define a loader for the testing data we can iterate through in 50-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )

    return train_loader, test_loader


def train(model, device, train_loader, optimizer, epoch, loss_criteria):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)

        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data)

        # Get the loss
        loss = loss_criteria(output, target)

        # Keep a running total
        train_loss += loss.item()

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Print metrics so we see some progress
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss


def test(model, device, test_loader, loss_criteria):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()
            #print(target)
            #print(predicted)

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print(
        'Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    # return average loss for the epoch
    return avg_loss