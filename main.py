#
# boostcamp AI Tech
# Image Classification Competition
#

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import os
import shutil
import time
import pickle
import multiprocessing

import pandas as pd

from loss import F1Loss
from model import MultiHeadClassifier
from dataset import ProfileClassEqualSplitTrainMaskDataset, EvalMaskDataset

def get_time() -> str:
    return time.strftime('%c', time.localtime(time.time()))

def clear_pycache(root: str = './') -> None:
    if os.path.exists(os.path.join(root, '__pycache__')):
        shutil.rmtree(os.path.join(root, '__pycache__'))

def clear_log_folders(root: str = './') -> None:
    if os.path.exists(os.path.join(root, 'checkpoints')):
        shutil.rmtree(os.path.join(root, 'checkpoints'))
    if os.path.exists(os.path.join(root, 'history')):
        shutil.rmtree(os.path.join(root, 'history'))
    if os.path.exists(os.path.join(root, 'results')):
        shutil.rmtree(os.path.join(root, 'results'))

def train_and_eval(done_epochs: int, train_epochs: int, clear_log: bool = False) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if clear_log:
        clear_log_folders()

    ######## Preparing Dataset ########
    print(f"Dataset | Data preparation start @ {get_time()}", flush=True)

    timestamp = get_time().replace(':', '')
    location = {
        'base_path': './dataset_fixed',
        'checkpoints_path': os.path.join('./checkpoints', timestamp),
        'history_path': os.path.join('./history', timestamp),
        'results_path': os.path.join('./results', timestamp)
    }
    os.makedirs(location['checkpoints_path'])
    os.makedirs(location['history_path'])
    os.makedirs(location['results_path'])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_train_val = ProfileClassEqualSplitTrainMaskDataset(
        data_dir=location['base_path'],
        transform=transform_train
    )
    dataset_test = EvalMaskDataset(
        data_dir=location['base_path'],
        transform=transform_test
    )

    dataset_train, dataset_val = dataset_train_val.split_dataset()

    batch_size = 16

    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        drop_last=False
    )

    train_batches = len(train_loader)
    val_batches = len(val_loader)
    test_batches = len(test_loader)

    ######## Model & Hyperparameters ########
    model = MultiHeadClassifier().to(device)

    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    plot_bound = 0

    ######## Loading Model ########
    if done_epochs > 0:
        checkpoint = torch.load(f"./checkpoints/epoch{done_epochs}.pt", map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        with open(f"./history/epoch{done_epochs}.pickle", 'rb') as fr:
            history = pickle.load(fr)
    else:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    ######## Train & Validation ########
    print('Train & Validation | Training start @ {}'.format(get_time()), flush=True)

    best_epoch = 0
    min_val_loss = 9999.
    for epoch in range(done_epochs, done_epochs + train_epochs):
        ######## Train ########
        print('Train | Epoch {:02d} start @ {}'.format(epoch + 1, get_time()), flush=True)

        model.train()
        train_loss = 0
        total = 0
        correct = 0

        for batch_index, (images, labels) in enumerate(train_loader):
            print('Train | Epoch {:02d} | Batch {} / {} start'.format(epoch + 1, batch_index + 1, train_batches), flush=True)

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Train loss
            train_loss += loss.item()

            # Train accuracy
            prediction = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

        train_acc = 100 * correct / total

        if (epoch + 1) > plot_bound:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

        print('Train | Loss: {:.4f} | Accuracy: {:.4f}%'.format(train_loss, train_acc), flush=True)

        # Save checkpoint
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(location['checkpoints_path'], f"epoch{epoch + 1}.pt"))

        ######## Validation ########
        print('Validation | Epoch {:02d} start @ {}'.format(epoch + 1, get_time()), flush=True)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            total = 0
            correct = 0

            for batch_index, (images, labels) in enumerate(val_loader):
                print('Validation | Epoch {:02d} | Batch {} / {} start'.format(epoch + 1, batch_index + 1, val_batches), flush=True)

                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Validation loss
                val_loss += loss.item()

                # Validation accuracy
                prediction = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (prediction == labels).sum().item()

            val_acc = 100 * correct / total

            if (epoch + 1) > plot_bound:
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

            print('Validation | Loss: {:.4f} | Accuracy: {:.4f}%'.format(val_loss, val_acc), flush=True)

            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_epoch = epoch + 1

        ######## Saving History ########
        with open(os.path.join(location['history_path'], f"epoch{epoch + 1}.pickle"), 'wb') as fw:
            pickle.dump(history, fw)

    print(f"Train & Validation | Finished training @ {get_time()}", flush=True)

    ######## Prediction Generation ########
    print(f"Prediction | Prediction start @ {get_time()}", flush=True)

    submission = pd.read_csv(os.path.join(location['base_path'], 'eval/info.csv'))
    predictions = []

    # Load best model selected by validation loss
    print(f"Prediction | Loading epoch {best_epoch} model (epoch with best validation loss)")
    if best_epoch != epoch + 1:
        checkpoint = torch.load(os.path.join(location['checkpoints_path'], f"epoch{best_epoch}.pt"), map_location=device)
        model.load_state_dict(checkpoint['model'])

    model.eval()
    with torch.no_grad():
        for batch_index, images in enumerate(test_loader):
            print('Prediction | Batch {} / {} start'.format(batch_index + 1, test_batches), flush=True)

            images = images.to(device)
            outputs = model(images)

            prediction = torch.argmax(outputs, dim=1)
            predictions.extend(prediction.cpu().numpy())

    # Save predictions
    submission['ans'] = predictions
    submission.to_csv(os.path.join(location['results_path'], 'submission.csv'), index=False)

    print(f"Prediction | Finished prediction @ {get_time()}", flush=True)

    ######## Learning Statistics ########
    if train_epochs == 0:
        epoch = done_epochs - 1

    plt.subplot(2, 1, 1)
    plt.plot(range(plot_bound + 1, epoch + 2), history['train_loss'], label='Train', color='red', linestyle='dashed')
    plt.plot(range(plot_bound + 1, epoch + 2), history['val_loss'], label='Validation', color='blue')

    plt.title('Loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(range(plot_bound + 1, epoch + 2), history['train_acc'], label='Train', color='red', linestyle='dashed')
    plt.plot(range(plot_bound + 1, epoch + 2), history['val_acc'], label='Validation', color='blue')

    plt.title('Accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(location['results_path'], 'result.png'), dpi=1000)

    print(f"Code execution done @ {get_time()}", flush=True)

if __name__ == '__main__':
    # Last checkpoint's training position
    done_epochs = 0

    # How much epochs to train now
    train_epochs = 30

    train_and_eval(done_epochs, train_epochs, clear_log=False)
