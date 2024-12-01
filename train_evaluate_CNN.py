from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from ConvNet import ConvNet 
import numpy as np 
import matplotlib.pyplot as plt

# Training data
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    losses = []
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        _, pred = torch.max(output, 1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    train_loss = np.mean(losses)
    train_acc = 100. * correct / total
    print(f'Train Epoch: {epoch} \tLoss: {train_loss:.4f} \tAccuracy: {correct}/{total} ({train_acc:.2f}%)\n')
    return train_loss, train_acc

# Testing data
def test(model, device, test_loader, criterion):
    model.eval()
    losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            losses.append(loss.item())

            # Calculate predictions
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    test_loss = np.mean(losses)
    test_acc = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({test_acc:.2f}%)\n')
    return test_loss, test_acc


def visualize_predictions(model, device, test_loader, classes, num_images=5):
    
    model.eval()
    images_displayed = 0

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    fig.suptitle("Test Images with Predicted Labels", fontsize=16)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Get model predictions
            output = model(data)
            _, preds = torch.max(output, 1)

            # Move tensors to CPU for plotting
            data = data.cpu()
            preds = preds.cpu()
            target = target.cpu()

            for i in range(data.size(0)):
                if images_displayed >= num_images:
                    break

                img = data[i]
                pred_label = classes[preds[i]]
                true_label = classes[target[i]]

                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                img = torch.clamp(img, 0, 1)  # Ensure the image is within [0,1]

                img_np = img.permute(1, 2, 0).numpy()

                axes[images_displayed].imshow(img_np)
                axes[images_displayed].set_title(f"P: {pred_label}\nT: {true_label}")
                axes[images_displayed].axis('off')

                images_displayed += 1

            if images_displayed >= num_images:
                break

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  
    plt.show()

def plot_loss(train_losses, test_losses, epochs):
    """
    Plot the training and test loss over epochs.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(train_accuracy, test_accuracy, num_epochs):
    """
    Plot test and train accuracy over epochs.
    """
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(15, 5))

    # Plot Autoencoder and CNN Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracy, label="Training Accuracy", color='blue')
    plt.plot(epochs, test_accuracy, label="Testing Accuracy", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy (%) Over Epochs")
    plt.legend()

    plt.show()


def run_main(mode=2, learning_rate=0.005, num_epochs=30, batch_size=64, log_dir='./logs'):
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected:", device)

    # Initialize the model and send to device 
    model = ConvNet(mode).to(device)

    # Define loss function.
    criterion = nn.CrossEntropyLoss()

    # Define optimizer function.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Define a learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Create transformations to apply to each data sample 
    transform = transforms.Compose([
        transforms.Resize((28, 28)),                   # Resize images to 28x28
        transforms.ToTensor(),                         # Convert PIL Image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])
    ])

    # Load datasets for training and testing
    train_path = './data/ASL/train'
    test_path = './data/ASL/test'

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training directory '{train_path}' does not exist.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Testing directory '{test_path}' does not exist.")

    train_dataset = ImageFolder(train_path, transform=transform)
    test_dataset = ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)

    best_accuracy = 0.0

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    # Save output into output.txt file 
    with open('output.txt', 'w') as f:
        # Run training for n_epochs specified in config 
        for epoch in range(1, num_epochs + 1):
            train_loss_val, train_accuracy_val = train(model, device, train_loader, optimizer, criterion, epoch)
            train_loss.append(train_loss_val)
            train_accuracy.append(train_accuracy_val)

            test_loss_val, test_accuracy_val = test(model, device, test_loader, criterion)
            test_loss.append(test_loss_val)
            test_accuracy.append(test_accuracy_val)

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/Train', train_loss_val, epoch)
            writer.add_scalar('Loss/Test', test_loss_val, epoch)
            writer.add_scalar('Accuracy/Train', train_accuracy_val, epoch)
            writer.add_scalar('Accuracy/Test', test_accuracy_val, epoch)

            # Write epoch results to output file
            f.write(f'Epoch {epoch}/{num_epochs}\n')
            f.write(f'Train set: Average loss: {train_loss_val:.4f}, Accuracy: {train_accuracy_val:.2f}%\n')
            f.write(f'Test set: Average loss: {test_loss_val:.4f}, Accuracy: {test_accuracy_val:.2f}%\n\n')

            # Check for improvement
            if test_accuracy_val > best_accuracy:
                best_accuracy = test_accuracy_val

            # Step the scheduler
            scheduler.step()

        # Write the best accuracy to the output file
        f.write(f"Best accuracy: {best_accuracy:.2f}%\n")

    writer.close()
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print("Training and evaluation finished")

    visualize_predictions(model, device, test_loader, train_dataset.classes, num_images=5)

    # Plot the loss and accuracy curves
    plot_loss(train_loss, test_loss, num_epochs)
    plot_accuracy(train_accuracy, test_accuracy, num_epochs)

if __name__ == '__main__':
    run_main()
