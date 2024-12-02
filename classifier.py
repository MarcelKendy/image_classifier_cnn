# Imports

# Torch: PyTorch library for deep learning operations.
# torchvision: For dataset and image transformations.
# numpy: For numerical operations.
# sklearn: For evaluation metrics and confusion matrix.
# matplotlib: For creating visualization charts
# sys: To write in a txt file
# time: To monitor the time spent
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights, VGG16_Weights, Inception_V3_Weights, DenseNet121_Weights, MobileNet_V2_Weights
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import sys 
import time

# AUX Class Logger to redirect stdout to both console and a file
class Logger:
    def __init__(self, filename="results.txt"):
        self.console = sys.stdout
        self.file = open(filename, "w")

    def write(self, message):
        self.console.write(message)  # Print to console
        self.file.write(message)    # Write to file
        self.file.flush()           # Ensure real-time writing

    def flush(self):
        self.console.flush()
        self.file.flush()

sys.stdout = Logger("results.txt")

# Functions

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20, patience=5, timeout=1*60):
    """
    Trains the model with early stopping, validation loss tracking, and optional timeout.

    Args:
        model: PyTorch model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: 'cpu' or 'cuda' (GPU).
        epochs: Maximum number of training epochs.
        patience: Early stopping patience.
        timeout: Maximum time for training in seconds. If False, no timeout is applied.

    Returns:
        trained model, train losses, val losses
    """
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    start_time = time.time()  # Start the timer

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Handle models that return tuples (Inception)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Handle models that return tuples (Inception)
                if isinstance(outputs, tuple):
                    outputs = outputs[0] 

                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")  # Save best model
        else:
            patience_counter += 1

        if patience_counter >= patience:  # Early stopping
            print(f"Early stopping at epoch {epoch + 1}")
            break

        elapsed_time = time.time() - start_time  # Time spent so far
        if timeout and elapsed_time > timeout:  # Check timeout if enabled
            print(f"Training stopped due to timeout at epoch {epoch + 1}.")
            break

    # Load the best model before returning
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))

    # Total time spent
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    timeout_minutes = timeout // 60 if timeout else "no limit"
    print(f"Time spent: {int(minutes)} min {int(seconds)} sec of {timeout_minutes} min allowed.")

    return model, train_losses, val_losses

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """
    Plots a confusion matrix using matplotlib.

    Args:
        cm (np.array): Confusion matrix.
        class_names (list): List of class names.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Function to evaluate model on test set and print metrics
def evaluate_model(model, test_loader, device, class_names, model_name):
    """
    Evaluate the model on the test set and compute performance metrics.
    
    Args:
        model: PyTorch model instance.
        test_loader: DataLoader for test data.
        device: 'cpu' or 'cuda' (GPU).
        class_names: List of class names.
        model_name: The name of the model 

    Returns:
        metrics: Dictionary containing metrics (accuracy, precision, recall, f1, confusion matrix).
    """
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    plot_confusion_matrix(cm, class_names, title=f"Confusion Matrix - {model_name}")

    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Main code
def main():
    print("Initializing program...")
    # Step 1/4: Dataset preparation
    print("Preparing Dataset...")
    DATASET_PATH = "image_dataset"
    IMAGE_SIZE = (299, 299)  # Reshaping resolution
    BATCH_SIZE = 32
    HOLD_OUT_SPLIT = 0.2  # 20% of the data will be reserved for the final test

    # Pre-Processing
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ])
    test_val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ])

    # Loading dataset and separating between training and testing (Hold-Out)
    dataset = datasets.ImageFolder(DATASET_PATH, transform=None)
    hold_out_size = int(HOLD_OUT_SPLIT * len(dataset))
    train_val_size = len(dataset) - hold_out_size

    train_val_dataset, test_dataset = random_split(dataset, [train_val_size, hold_out_size])

    # Dividing training/testing from the remaining data
    train_size = int(0.7 * len(train_val_dataset))  # 70% for training
    val_size = len(train_val_dataset) - train_size  # The rest for validation
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    # Aplying transformations
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_val_transform
    test_dataset.dataset.transform = test_val_transform

    # Creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = dataset.classes

    # Step 2/4: Models initialization
    print("Initializing models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Checking which device to use

    # ResNet18 model
    print("ResNet18...")
    model_resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model_resnet.parameters():
        param.requires_grad = False
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, len(class_names))
    model_resnet = model_resnet.to(device)

    # VGG16 model
    print("VGG16...")
    model_vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    for param in model_vgg.parameters():
        param.requires_grad = False
    model_vgg.classifier[6] = nn.Linear(model_vgg.classifier[6].in_features, len(class_names))
    model_vgg = model_vgg.to(device)

    # Inception_v3 model
    print("Inception_v3...")
    model_inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    for param in model_inception.parameters():
        param.requires_grad = False
    model_inception.AuxLogits.fc = nn.Linear(model_inception.AuxLogits.fc.in_features, len(class_names))
    model_inception.fc = nn.Linear(model_inception.fc.in_features, len(class_names))
    model_inception = model_inception.to(device)


    # DenseNet model
    print("DenseNet...")
    model_densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    for param in model_densenet.parameters():
        param.requires_grad = False
    model_densenet.classifier = nn.Linear(model_densenet.classifier.in_features, len(class_names))
    model_densenet = model_densenet.to(device)

    # MobileNetV2 model
    print("MobileNetV2...")
    model_mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model_mobilenet.parameters():
        param.requires_grad = False
    model_mobilenet.classifier[1] = nn.Linear(model_mobilenet.classifier[1].in_features, len(class_names))
    model_mobilenet = model_mobilenet.to(device)

    # Step 3/4: Training models
    print("Training models...")
    print("Training ResNet18...")
    optimizer_resnet = optim.Adam(model_resnet.parameters(), lr=1e-3)
    model_resnet, train_losses_resnet, val_losses_resnet = train_model(model_resnet, train_loader, val_loader,
                                                                       nn.CrossEntropyLoss(), optimizer_resnet, device)

    print("Training VGG16...")
    optimizer_vgg = optim.Adam(model_vgg.parameters(), lr=1e-3)
    model_vgg, train_losses_vgg, val_losses_vgg = train_model(model_vgg, train_loader, val_loader,
                                                              nn.CrossEntropyLoss(), optimizer_vgg, device)

    print("Training InceptionV3...")
    optimizer_inception = optim.Adam(model_inception.parameters(), lr=1e-3)
    model_inception, train_losses_inception, val_losses_inception = train_model(model_inception, train_loader, val_loader,
                                                                                nn.CrossEntropyLoss(), optimizer_inception, device)

    print("Training DenseNet...")
    optimizer_densenet = optim.Adam(model_densenet.parameters(), lr=1e-3)
    model_densenet, train_losses_densenet, val_losses_densenet = train_model(model_densenet, train_loader, val_loader,
                                                                             nn.CrossEntropyLoss(), optimizer_densenet, device)

    print("Training MobileNetV2...")
    optimizer_mobilenet = optim.Adam(model_mobilenet.parameters(), lr=1e-3)
    model_mobilenet, train_losses_mobilenet, val_losses_mobilenet = train_model(model_mobilenet, train_loader, val_loader,
                                                                                nn.CrossEntropyLoss(), optimizer_mobilenet, device)

    # Step 4/4: Evaluate models
    print("Evaluating models...")
    print("Evaluating ResNet18 on Test Data...")
    metrics_resnet = evaluate_model(model_resnet, test_loader, device, class_names, "ResNet18")

    print("Evaluating VGG16 on Test Data...")
    metrics_vgg = evaluate_model(model_vgg, test_loader, device, class_names, "VGG16")

    print("Evaluating InceptionV3 on Test Data...")
    metrics_inception = evaluate_model(model_inception, test_loader, device, class_names, "InceptionV3")

    print("Evaluating DenseNet on Test Data...")
    metrics_densenet = evaluate_model(model_densenet, test_loader, device, class_names, "DenseNet")

    print("Evaluating MobileNetV2 on Test Data...")
    metrics_mobilenet = evaluate_model(model_mobilenet, test_loader, device, class_names, "MobileNetV2")

    return metrics_resnet, metrics_vgg, metrics_inception, metrics_densenet, metrics_mobilenet

if __name__ == "__main__":
    metrics_resnet, metrics_vgg, metrics_inception, metrics_densenet, metrics_mobilenet = main()
