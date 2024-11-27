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

def train(model, device, train_loader, optimizer, criterion, epoch):
    '''
    Trains the model for one epoch and optimizes it.
    '''
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

        # Calculate predictions
        _, pred = torch.max(output, 1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    train_loss = np.mean(losses)
    train_acc = 100. * correct / total
    print(f'Train Epoch: {epoch} \tLoss: {train_loss:.4f} \tAccuracy: {correct}/{total} ({train_acc:.2f}%)\n')
    return train_loss, train_acc

def test(model, device, test_loader, criterion):
    '''
    Evaluates the model on the test dataset.
    '''
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

def visualize_samples(train_loader, classes):
    '''
    Visualizes a few samples from the training dataset to verify data loading.

    Parameters:
    - train_loader: DataLoader for training data.
    - classes: List of class names.
    '''
    images, labels = next(iter(train_loader))
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for idx, ax in enumerate(axes):
        ax.imshow(images[idx].permute(1, 2, 0).cpu().numpy())
        ax.set_title(f"Label: {classes[labels[idx]]}")
        ax.axis('off')
    plt.show()

def run_main(FLAGS):
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected:", device)

    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)

    # Define loss function.
    criterion = nn.CrossEntropyLoss()

    # Define optimizer function.
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)

    # Define a learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Create transformations to apply to each data sample 
    transform = transforms.Compose([
        transforms.Resize((28, 28)),                   # Resize images to 28x28
        transforms.ToTensor(),                         # Convert PIL Image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize for RGB images
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

    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # Visualize sample images (optional)
    visualize_samples(train_loader, train_dataset.classes)

    # Initialize TensorBoard writer
    writer = SummaryWriter(FLAGS.log_dir)

    best_accuracy = 0.0
    patience = 10  # For early stopping
    trigger_times = 0

    # Save output into output.txt file 
    with open('output.txt', 'w') as f:
        # Run training for n_epochs specified in config 
        for epoch in range(1, FLAGS.num_epochs + 1):
            train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion, epoch)
            test_loss, test_accuracy = test(model, device, test_loader, criterion)

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Test', test_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/Test', test_accuracy, epoch)

            # Write epoch results to output file
            f.write(f'Epoch {epoch}/{FLAGS.num_epochs}\n')
            f.write(f'Train set: Average loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%\n')
            f.write(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%\n\n')

            # Check for improvement
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
                trigger_times = 0
            else:
                trigger_times += 1
                print(f"No improvement in accuracy for {trigger_times} epoch(s).")
                if trigger_times >= patience:
                    print("Early stopping triggered.")
                    break

            # Step the scheduler
            scheduler.step()

        # Write the best accuracy to the output file
        f.write(f"Best accuracy: {best_accuracy:.2f}%\n")

    writer.close()
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print("Training and evaluation finished")

if __name__ == '__main__':
    # Set parameters for CNN training
    parser = argparse.ArgumentParser(description='Train ConvNet on ASL Letters.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        choices=[1, 2],
                        help='Select mode between 1-2.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int, default=60,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put TensorBoard logs.')

    FLAGS = parser.parse_args()

    run_main(FLAGS)