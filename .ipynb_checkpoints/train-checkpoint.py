import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score

from model import CAD_MENBI_Classifier
from dataset import CAD_MENBI_Dataset
from utils import get_transforms, get_device


def train():
    logging.basicConfig(filename='logs/training_resnet152.log', level=logging.INFO, 
            format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        
    # 图像预处理
    train_transform = get_transforms(train=True)

    dataset = CAD_MENBI_Dataset(root_dir='data_processed', transform=train_transform)

    device = get_device()
    n_samples = len(dataset)
    print(n_samples)
    n_classes = 4
    epochs = 100
    batch_size = 32
    num_workers = 8

    # Compute class weights
    targets = np.array(dataset.labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
    class_weights = torch.FloatTensor(class_weights).cuda()
    print(class_weights)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(kf.split(range(n_samples))):
        print(f"Fold {fold + 1}")
        logging.info(f"Fold {fold + 1}")

        # Initialize Gastritis Classifier for this part
        model = CAD_MENBI_Classifier(n_classes).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Training loop
        for epoch in range(epochs):
            model.train()
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                images, labels = images.cuda(), labels.cuda()

                optimizer.zero_grad()
                # Resnet
                outputs = model(images)
                _, predicted_train = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

            accuracy_train = correct_train / total_train

            # Validation loop
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                all_labels = []
                all_preds = []
                for images, labels in val_loader:
                    images, labels = images.cuda(), labels.cuda()
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
            
                accuracy_val = correct / total
                balanced_accuracy_val = balanced_accuracy_score(all_labels, all_preds)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Train Acc: {accuracy_train * 100:.2f}%, Val Acc: {accuracy_val * 100:.2f}%, Bal Val Acc: {balanced_accuracy_val * 100:.2f}%')
                logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Train Acc: {accuracy_train * 100:.2f}%, Val Acc: {accuracy_val * 100:.2f}%, Bal Val Acc: {balanced_accuracy_val * 100:.2f}%')

            if (epoch+1) % 20 == 0:
                # Save weights for each fold and epoch
                os.makedirs(f"weights/", exist_ok=True)
                torch.save(model.state_dict(), f'weights/fold_{fold+1}_epoch_{epoch+1}.pth')


if __name__ == '__main__':
    train()