from __future__ import print_function  # Ensures compatibility of print() in Python 2.x
import torch
import torch.nn as nn  # Provides neural network layers
import torch.nn.functional as F  # Provides activation functions and other operations
import torch.optim as optim  # Provides optimization algorithms
from torchvision import datasets, transforms  # For loading datasets and applying transformations
from torch.optim.lr_scheduler import StepLR  # For scheduling the learning rate

# -------------------------------
# Define the Neural Network Model
# -------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer: takes a 1-channel (grayscale) image, produces 32 channels
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # Second convolutional layer: takes 32 channels, produces 64 channels
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout layers help prevent overfitting by randomly zeroing out some activations
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # Fully connected layer: input size 9216 (from the flattened feature maps) to 128 features
        self.fc1 = nn.Linear(9216, 128)
        # Output layer: maps 128 features to 10 output classes (digits 0-9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First convolution and ReLU activation
        x = self.conv1(x)
        x = F.relu(x)
        # Second convolution
        x = self.conv2(x)
        # Apply max pooling with a 2x2 window to reduce spatial dimensions
        x = F.max_pool2d(x, 2)
        # Apply dropout to reduce overfitting
        x = self.dropout1(x)
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)
        # Fully connected layer with ReLU activation
        x = self.fc1(x)
        x = F.relu(x)
        # Second dropout layer
        x = self.dropout2(x)
        # Final output layer
        x = self.fc2(x)
        # Apply log softmax to obtain log-probabilities for each class
        output = F.log_softmax(x, dim=1)
        return output

# -------------------------------
# Define the Training Function
# -------------------------------
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move data to the device (GPU or CPU)
        optimizer.zero_grad()  # Clear gradients from the previous step
        output = model(data)  # Forward pass: compute predictions
        loss = F.nll_loss(output, target)  # Compute the negative log likelihood loss
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model parameters
        # Print training progress at specified intervals
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# -------------------------------
# Define the Testing Function
# -------------------------------
def test(model, device, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data to the device
            output = model(data)  # Forward pass: compute predictions
            # Sum up the batch loss (using reduction='sum' to aggregate losses)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # Get the predicted class with the highest probability
            pred = output.argmax(dim=1, keepdim=True)
            # Count correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)  # Average loss over the test set
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy

# -------------------------------
# Main Function (without command-line parsing)
# -------------------------------
def main():
    # Fixed hyperparameters
    batch_size = 64
    test_batch_size = 1000
    max_epochs = 100         # Maximum number of epochs (upper bound)
    lr = 1.0                 # Learning rate
    gamma = 0.7              # Learning rate step gamma
    seed = 1                 # Random seed for reproducibility
    log_interval = 10        # Interval for logging training progress
    
    # Early stopping parameters
    target_accuracy = 95.0   # Stop training if accuracy reaches or exceeds 95%
    patience = 4             # Number of epochs to wait without significant improvement before stopping
    min_delta = 0.05          # Minimum improvement (in percentage points) to reset patience
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Determine if CUDA is available and set device accordingly
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # Prepare the training data with augmentation and normalization
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomAffine(
                               degrees=30, translate=(0.5, 0.5), scale=(0.25, 1),
                               shear=(-30, 30, -30, 30)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    # Prepare the testing data (only normalization)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    # Initialize the model, optimizer, and learning rate scheduler
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    
    best_accuracy = 0.0
    no_improvement_count = 0

    # Training loop with early stopping
    for epoch in range(1, max_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        _, accuracy = test(model, device, test_loader)
        scheduler.step()
        
        # Check if target accuracy is reached
        if accuracy >= target_accuracy:
            print(f"Target accuracy of {target_accuracy}% reached (accuracy: {accuracy:.2f}%). Stopping training.")
            break
        
        # Check for significant improvement
        if accuracy - best_accuracy >= min_delta:
            best_accuracy = accuracy
            no_improvement_count = 0  # Reset patience counter if improvement is significant
        else:
            no_improvement_count += 1
            print(f"No significant improvement for {no_improvement_count} epoch(s).")
        
        # If no significant improvement for a number of epochs, stop training
        if no_improvement_count >= patience:
            print(f"No significant improvement in the last {patience} epochs. Early stopping.")
            break

    # Save the trained model's state dictionary
    torch.save(model.state_dict(), "pytorch_model.pt")
    print("Model saved to pytorch_model.pt")

if __name__ == '__main__':
    main()  # Run the main function
