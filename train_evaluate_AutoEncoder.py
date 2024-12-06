import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from AutoEncoder import AutoEncoder
from AutoEncoderClassifier import AutoEncoderClassifier
import matplotlib.pyplot as plt

# This function will train the regular Autoencoder.
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        # Forward pass
        reconstructed_output, _ = model(data)

        # Compute reconstruction loss
        loss = criterion(reconstructed_output, data)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}, Average Training Loss: {avg_loss:.4f}")

    return avg_loss

# This function will perform training for the CNN
# with the Autoencoder as input.
def train_with_autoencoder(autoencoder, cnn, device, train_loader, optimizer, criterion, epoch):
    autoencoder.eval()  # Fix autoencoder weights
    cnn.train()
    total_loss = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Extract latent representations from autoencoder
        with torch.no_grad():
            _, latent_representation = autoencoder(data)

        # Forward pass through CNN
        output = cnn(latent_representation)

        # Compute classification loss
        loss = criterion(output, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Autoencoder CNN Loss: {avg_loss:.4f}")
    return avg_loss

# This function will test the Autoencoder
def test(model, device, test_loader, criterion):
    model.eval()
    losses = []

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)

            # Forward pass
            reconstructed_output, _ = model(data)

            # Compute reconstruction loss
            loss = criterion(reconstructed_output, data)
            losses.append(loss.item())

    test_loss = sum(losses) / len(losses)
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')
    return test_loss

# This function will test the CNN
# with the autoencoder as input.
def test_with_autoencoder(autoencoder, cnn, device, test_loader, criterion):
    autoencoder.eval()
    cnn.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            # Extract latent representations from autoencoder
            _, latent_representation = autoencoder(data)

            # Forward pass through CNN
            output = cnn(latent_representation)

            # Compute loss
            loss = criterion(output, labels)
            total_loss += loss.item()

            # Compute accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    return avg_loss, accuracy

def denormalize(tensor, mean, std):
    # Denormalize the tensor by reversing the normalization operation
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Multiply by std and add mean
    return tensor

def visualize_samples(model, device, test_loader):
    # Visualize a few test samples and their reconstructions.
    model.eval()
    images, _ = next(iter(test_loader))
    images = images.to(device)

    # Denormalize the images before visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    images = denormalize(images, mean, std)

    
    with torch.no_grad():
        reconstructed, _ = model(images) 

    # Create a figure with 2 columns: one for originals, one for reconstructions
    fig, axes = plt.subplots(5, 2, figsize=(12, 15))  # 5 rows, 2 columns
    for idx in range(5):
        # Display original image
        ax = axes[idx, 0]
        ax.imshow(images[idx].cpu().permute(1, 2, 0).clamp(0,1))
        ax.set_title("Original")
        ax.axis('off')

        # Display reconstructed image
        ax = axes[idx, 1]
        ax.imshow(reconstructed[idx].cpu().permute(1, 2, 0).clamp(0,1))
        ax.set_title("Reconstructed")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_loss(train_losses, test_losses):
    # Plot the training and test loss over epochs.
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, test_losses, label='Test Loss', color='red')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_metrics(autoencoder_loss, cnn_loss, cnn_accuracy, num_epochs):
    
    # Plot Autoencoder loss, CNN loss, and CNN accuracy.
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(15, 5))

    # Plot Autoencoder and CNN Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, autoencoder_loss, label="Autoencoder Loss")
    plt.plot(epochs, cnn_loss, label="CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss During Training")
    plt.legend()

    # Plot CNN Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, cnn_accuracy, label="CNN Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CNN Accuracy During Training")
    plt.legend()

    plt.tight_layout()
    plt.show()


def run_main(batch_size=64, num_epochs=30, learning_rate=0.001, data_dir='./data/ASL', log_dir='./logs'):
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected:", device)

    # Initialize the autoencoder model
    model = AutoEncoder().to(device)
    cnn = AutoEncoderClassifier().to(device)

    # Define the loss function (MSELoss for autoencoders, Cross Entropy for CNN)
    criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()


    # Define the optimizer (Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cnn_optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    # Create transformations to apply to each data sample
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),        # Convert PIL Image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize RGB
    ])

    # Load datasets for training and testing
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    # Initialize TensorBoard writer (optional)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)

    # Initialize lists to store losses
    train_losses = []
    test_losses = []
    cnn_train_losses = []
    cnn_test_losses = []
    cnn_accuracy = []

    # Run training and testing
    best_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        # Training step
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        train_losses.append(train_loss)

        # Testing step
        test_loss = test(model, device, test_loader, criterion)
        test_losses.append(test_loss)

        # Train CNN with latent representation
        cnn_train_loss = train_with_autoencoder(model, cnn, device, train_loader, cnn_optimizer, classification_criterion, epoch)
        cnn_train_losses.append(cnn_train_loss)

        # Testing CNN
        cnn_test_loss, cnn_test_accuracy = test_with_autoencoder(model, cnn, device, test_loader, classification_criterion)
        cnn_accuracy.append(cnn_test_accuracy)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)

        # Save the best autoencoder model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'best_autoencoder_model.pth')

    # Visualize samples and their reconstructions (optional)
    visualize_samples(model, device, test_loader)

    # Plot the loss and accuracy curves
    plot_loss(train_losses, test_losses)
    plot_metrics(train_losses, cnn_train_losses, cnn_accuracy, num_epochs)


if __name__ == '__main__':
    run_main()
