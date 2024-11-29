import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """
    Plot a confusion matrix using matplotlib.
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
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Training function with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=15, patience=5):
    """
    Trains the model with early stopping and validation loss tracking.

    Args:
        model: PyTorch model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: 'cpu' or 'cuda' (GPU).
        epochs: Maximum number of training epochs.
        patience: Early stopping patience.

    Returns:
        trained model, train losses, val losses
    """
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")  # Save best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    model.load_state_dict(torch.load("best_model.pth"))
    return model, train_losses, val_losses

# Function to evaluate model on test set and print metrics
def evaluate_model(model, test_loader, device, class_names, model_name):
    """
    Evaluate the model on the test set and compute performance metrics.
    
    Args:
        model: PyTorch model instance.
        test_loader: DataLoader for test data.
        device: 'cpu' or 'cuda' (GPU).
        class_names: List of class names.

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
    # Step 1: Dataset preparation
    DATASET_PATH = "image_dataset"
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = dataset.classes

    # Step 2: Model initialization (two different models)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ResNet18 model
    model_resnet = models.resnet18(pretrained=True)
    for param in model_resnet.parameters():
        param.requires_grad = False
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, len(class_names))
    model_resnet = model_resnet.to(device)

    # VGG16 model
    model_vgg = models.vgg16(pretrained=True)
    for param in model_vgg.parameters():
        param.requires_grad = False
    model_vgg.classifier[6] = nn.Linear(model_vgg.classifier[6].in_features, len(class_names))
    model_vgg = model_vgg.to(device)

    # Step 3: Training ResNet18
    optimizer_resnet = optim.Adam(model_resnet.parameters(), lr=1e-3)
    model_resnet, train_losses_resnet, val_losses_resnet = train_model(model_resnet, train_loader, val_loader,
                                                                      nn.CrossEntropyLoss(), optimizer_resnet, device)

    # Step 4: Training VGG16
    optimizer_vgg = optim.Adam(model_vgg.parameters(), lr=1e-3)
    model_vgg, train_losses_vgg, val_losses_vgg = train_model(model_vgg, train_loader, val_loader,
                                                              nn.CrossEntropyLoss(), optimizer_vgg, device)

    # Step 5: Evaluate both models on test data
    print("\nEvaluating ResNet18 on Test Data:")
    metrics_resnet = evaluate_model(model_resnet, test_loader, device, class_names, "ResNet18")

    print("\nEvaluating VGG16 on Test Data:")
    metrics_vgg = evaluate_model(model_vgg, test_loader, device, class_names, "VGG16")

    return metrics_resnet, metrics_vgg


if __name__ == "__main__":
    metrics_resnet, metrics_vgg = main()
