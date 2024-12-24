# Imports

# Torch: PyTorch library for deep learning operations.
# torchvision: For dataset and image transformations.
# sklearn: For evaluation metrics and confusion matrix.
# matplotlib: For creating visualization charts
# seaborn: Data visualization to create the confusion matrix.
# sys: To write in a txt file
# time: To monitor the time spent
# random: To randomize subsets of configs of hyperparameters
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights, VGG16_Weights, Inception_V3_Weights, DenseNet121_Weights, MobileNet_V2_Weights
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys 
import time
import random

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
def train_model(model, train_loader, val_loader, criterion, optimizer, device, model_name, epochs=20, patience=5, timeout=1200*60):
    """
    Trains the model with early stopping, validation loss tracking, and optional timeout.

    Args:
        model: PyTorch model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: 'cpu' or 'cuda' (GPU).
        model_name: Name of the model (used for saving graphs).
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

    # Save validation loss graph
    save_loss_graph(train_losses, val_losses, model_name, f"loss_progressions/{model_name}_loss.png")

    return model, train_losses, val_losses

# Function to save the Training and Validation Loss Progression
def save_loss_graph(train_losses, val_losses, model_name, save_path=None):
    """
    Save the training and validation loss graph during training.
    
    Args:
        train_losses: List of training losses for each epoch.
        val_losses: List of validation losses for each epoch.
        model_name: Name of the model.
        save_path: Path to save the plot image. If None, the plot will be shown on the screen.
    """
    plt.figure()
    # Plot training losses
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label="Training Loss")
    # Plot validation losses
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', label="Validation Loss")
    
    # Add titles and labels
    plt.title(f"Loss Progression - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()  # Add legend to distinguish the two lines

    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the plot to avoid displaying it
    else:
        plt.show()

# Function to plot the Confusion Matrix
def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    """
    Plot and save the confusion matrix.

    Args:
        cm: Confusion matrix.
        class_names: List of class names.
        title: Title of the plot.
        save_path: Path to save the plot image. If None, the plot will be shown on the screen.
    """
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the plot to avoid displaying it
    else:
        plt.show()

# Function to evaluate model on test set and print metrics
def evaluate_model(model, test_loader, device, class_names, model_name):
    """
    Evaluate the model on the test set and compute performance metrics.
    Logs detailed classification results for each image in a separate file.

    Args:
        model: PyTorch model instance.
        test_loader: DataLoader for test data.
        device: 'cpu' or 'cuda' (GPU).
        class_names: List of class names.
        model_name: Name of the model.

    Returns:
        metrics: Dictionary containing metrics (accuracy, precision, recall, f1, confusion matrix).
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    log_file = f"classifications_logs/classifications_log_{model_name}.txt"

    with open(log_file, "w") as log:
        log.write(f"Classification Results for Model: {model_name}\n")
        log.write("=" * 50 + "\n")

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Get probabilities (softmax) and predictions
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                # Log classification results for each image in the batch
                for idx in range(inputs.size(0)):
                    image_name = f"Image_{i * test_loader.batch_size + idx + 1}"  # Generate an image name
                    label = class_names[labels[idx].item()]
                    prediction = class_names[preds[idx].item()]
                    probs = probabilities[idx].cpu().numpy()

                    log.write(f"{image_name}\n")
                    log.write(f"True Label: {label}\n")
                    log.write(f"Predicted: {prediction}\n")
                    log.write(f"Probabilities: {', '.join(f'{class_names[j]}: {probs[j]:.4f}' for j in range(len(class_names)))}\n")
                    log.write("-" * 50 + "\n")

    # Compute metrics
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

    # Save confusion matrix plot
    plot_confusion_matrix(cm, class_names, title=f"Confusion Matrix - {model_name}", save_path=f"confusion_matrices/{model_name}_confusion_matrix.png")

    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Function to initialize models
def initialize_model(model_name, num_classes, freeze_features=False):
    """
    Initializes and customizes a deep learning model for a classification task.

    Args:
        model_name: The name of the model to initialize. Options include:
            - "ResNet18"
            - "VGG16"
            - "InceptionV3"
            - "DenseNet"
            - "MobileNetV2"
        num_classes: The number of output classes for the classification task.
        freeze_features: Whether to freeze the feature extraction layers of the model 
                         (default: False).

    Returns:
        A PyTorch model customized for the specified number of classes, with optional 
        frozen features.
    """
    available_models = ["ResNet18", "VGG16", "InceptionV3", "DenseNet", "MobileNetV2"]
    if (model_name not in available_models):
        print(f"Model {model_name} not available. Models available: {available_models}")
        return False
    if model_name == "ResNet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "VGG16":
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "InceptionV3":
        model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "DenseNet":
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Freezing features 
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
        # Ensuring classification layers remain trainable (I guess...)
        if isinstance(model, models.ResNet):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif isinstance(model, models.VGG):
            for param in model.classifier[6].parameters():
                param.requires_grad = True
        elif isinstance(model, models.Inception3):
            # Inception V3 has 'fc' and 'AuxLogits.fc' layers
            for param in model.fc.parameters():
                param.requires_grad = True
            for param in model.AuxLogits.fc.parameters():
                param.requires_grad = True
        elif isinstance(model, models.DenseNet):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif isinstance(model, models.MobileNetV2):
            for param in model.classifier[1].parameters():
                param.requires_grad = True
    
    return model

# Function for hyperparameter optimization
def hyperparameter_optimization(hyperparams_dict, train_dataset, val_dataset, device, class_names, num_epochs=5, combinations_percentage=0.6, upper_limit_testings=8, learning_rates=[1e-2, 1e-3, 1e-4], batch_sizes=[8, 16, 32], optimizers=[optim.Adam, optim.SGD]):
    """
    Optimizes hyperparameters for each model using random search.

    Args:
        hyperparams_dict: Dictionary of hyperparameters for all models.
        train_dataset: Dataset for training.
        val_dataset: Dataset for validation.
        device: 'cpu' or 'cuda' (GPU).
        class_names: List of class names.
        num_epochs: Number of epochs to train for each parameter combination (default: 5).
        combinations_percentage: Percentage of all possible hyperparameter combinations to test (default: 0.6).
        upper_limit_testings: Maximum number of hyperparameter combinations to test, regardless of the percentage (default: 8).
        learning_rates: List of possible learning rates to use during optimization (default: [1e-2, 1e-3, 1e-4]).
        batch_sizes: List of possible batch sizes to use during optimization (default: [8, 16, 32]).
        optimizers: List of possible optimizer classes to use during optimization (default: [optim.Adam, optim.SGD]).

    Updates:
        hyperparams_dict is updated in-place with the optimized hyperparameters.
    """
    # Possible values for random search defined on params          
    n_testings = int(combinations_percentage * (len(learning_rates) * len(batch_sizes) * len(optimizers)))  # % of all random combinations possible based on param      
    max_n_testings = min(n_testings, upper_limit_testings) # Upper limit to the number of random combinations

    for model_name, params in hyperparams_dict.items():
        print(f"Optimizing hyperparameters for {model_name}...")

        best_val_loss = float('inf')
        best_params = params
        tried_combinations = set()

        while len(tried_combinations) < max_n_testings:
            # Randomly select a combination
            lr = random.choice(learning_rates)
            batch_size = random.choice(batch_sizes)
            optimizer_class = random.choice(optimizers)

            current_combination = (lr, batch_size, optimizer_class)
            if current_combination in tried_combinations:
                continue
            tried_combinations.add(current_combination)

            print(f"Testing combination ({len(tried_combinations)}/{max_n_testings}): Learning Rate = {lr}, Batch Size = {batch_size}, Optimizer = {optimizer_class.__name__}")

            # DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Initialize model, optimizer, and criterion
            model = initialize_model(model_name, len(class_names)).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optimizer_class(model.parameters(), lr=lr)

            # Train for the specified number of epochs
            for epoch in range(num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    model.zero_grad()
                    optimizer.zero_grad()

                    outputs = model(inputs)
                    if isinstance(outputs, tuple):  
                        outputs = outputs.logits

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # Validate and calculate validation loss
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        if isinstance(outputs, tuple):
                            outputs = outputs.logits
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

            # Update best parameters if the current loss is lower
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {"learning_rate": lr, "batch_size": batch_size, "optimizer": optimizer_class}

            # Clean up to prevent memory issues
            del model
            del optimizer
            torch.cuda.empty_cache()

        # Save the best hyperparameters for this model
        hyperparams_dict[model_name] = best_params
        print(f"Best params for {model_name} (Val Loss = {best_val_loss}): {best_params}")

# Main code
def main():
    print("Initializing program...")

    # Step 1/4: Dataset preparation
    print("Preparing Dataset...")
    DATASET_PATH = "image_dataset"
    IMAGE_SIZE = (299, 299)  # Reshaping resolution
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
    class_names = dataset.classes
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Checking which device to use
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

    # Default hyperparameters for all models
    default_hyperparams = {
        "batch_size": 16,
        "learning_rate": 1e-4,
        "optimizer": optim.Adam
    }

    # Hyperparameter dictionary for all models
    hyperparams_dict = {
        "ResNet18": default_hyperparams.copy(),
        "VGG16": default_hyperparams.copy(),
        "InceptionV3": default_hyperparams.copy(),
        "DenseNet": default_hyperparams.copy(),
        "MobileNetV2": default_hyperparams.copy()
    }

    # Hyperparameter optimization
    print("Optimizing hyperparameters...")
    hyperparameter_optimization(hyperparams_dict, train_dataset, val_dataset, device, class_names) # If you comment this line, the hyperparams_dict will remain unchanged with the default values.

    # Step 2/4: Models initialization
    print("Initializing models...")

    # Initialize models_list, a tuple with the model and its name
    models_list = [(None, "ResNet18"), (None, "VGG16"), (None, "InceptionV3"), (None, "DenseNet"), (None, "MobileNetV2")]

    # Iterate through models_list to initialize the models
    for i, (model, model_name) in enumerate(models_list):
        print(f"Initializing {model_name}...")
        initialized_model = initialize_model(model_name, num_classes=len(class_names), freeze_features=True).to(device)
        models_list[i] = (initialized_model, model_name)  # Update the "None" placeholder with the initialized model

    # Step 3/4: Training models with selected hyperparameters
    print("Training models...")

    # Iterate over the list of models and train them with the apropriate hyperparams
    for model, model_name in models_list:
        batch_size = hyperparams_dict[model_name]["batch_size"]
        learning_rate = hyperparams_dict[model_name]["learning_rate"]
        optimizer_type = hyperparams_dict[model_name]["optimizer"]
        print(f"Training {model_name}... (Hyper Params: Learning Rate = {learning_rate}, Batch Size = {batch_size}, Optimizer = {optimizer_type.__name__})")

        # Create DataLoader with the hyperparameter batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Set optimizer with the selected learning rate
        optimizer = optimizer_type(model.parameters(), lr=learning_rate)
        
        # Train the model
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer, device, model_name)

    # Step 4/4: Evaluate models
    print("Evaluating models...") 
    metrics = {}

    # Iterate over the list of models and evaluate them on test data
    for model, model_name in models_list:
        print(f"Evaluating {model_name} on Test Data...")
        metrics[model_name] = evaluate_model(model, test_loader, device, class_names, model_name)
    return metrics

if __name__ == "__main__":
    metrics = main()

