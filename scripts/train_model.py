# in train_model.py, we train a simple 1D CNN model to classify breathing windows into Normal, Hypopnea and Apnea
# we also use LOPO cross-validation for evaluation

# first, we import the libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from models.cnn_model import SimpleCNN


# this function loads preprocessed datasets with window features
def load_participant_data(dataset_dir, participant_id):
    X = np.load(os.path.join(dataset_dir, f"{participant_id}_X.npy"))
    y = np.load(os.path.join(dataset_dir, f"{participant_id}_y.npy"))
    return X, y


# this function is used to train the CNN on training participants and evaluate the testing participant
def train_and_evaluate(train_X, train_y, test_X, test_y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    # we handle class imbalance and assign higher weights to rare classes
    class_counts = np.bincount(train_y)
    class_weights = 1.0 / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_X = torch.tensor(train_X, dtype=torch.float32).unsqueeze(1).to(device)
    train_y = torch.tensor(train_y, dtype=torch.long).to(device)
    test_X = torch.tensor(test_X, dtype=torch.float32).unsqueeze(1).to(device)
    test_y = torch.tensor(test_y, dtype=torch.long).to(device)

    # we train and evaluate the model for a few fixed number of epochs
    epochs = 10
    for epoch in range(epochs):
        model.train()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        outputs = model(test_X)
        preds = torch.argmax(outputs, dim=1)
    preds = preds.cpu().numpy()
    test_y = test_y.cpu().numpy()
    # we compute the classification metrics (as required)
    acc = accuracy_score(test_y, preds)
    prec = precision_score(test_y, preds, average='macro', zero_division=0)
    rec = recall_score(test_y, preds, average='macro', zero_division=0)
    cm = confusion_matrix(test_y, preds)
    return acc, prec, rec, cm


# the main function
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "Dataset")
    participants = ["AP01", "AP02", "AP03", "AP04", "AP05"]
    for test_id in participants:
        print("\nThe participant that is being tested is:", test_id)
        train_X_all = []
        train_y_all = []
        for pid in participants:
            X, y = load_participant_data(dataset_dir, pid)
            if pid == test_id:
                test_X, test_y = X, y
            else:
                train_X_all.append(X)
                train_y_all.append(y)
        train_X = np.concatenate(train_X_all)
        train_y = np.concatenate(train_y_all)
        acc, prec, rec, cm = train_and_evaluate(train_X, train_y, test_X, test_y)
        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    main()