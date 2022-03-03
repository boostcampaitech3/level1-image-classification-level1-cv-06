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
import math
import time
import pickle
import multiprocessing

import pandas as pd

from loss import F1Loss
from model import MultiHeadClassifier, ResNet_Ensemble, ResNet_Ensemble2, ResNet_Ensemble_debugged
from dataset import ProfileClassEqualSplitTrainMaskDataset, EvalMaskDataset
import numpy as np
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
    print(f"Dataset | Data preparation start @ {get_time()}")

    timestamp = get_time().replace(':', '')
    location = {
        'base_path': './dataset_fixed',
        'checkpoints_path': './checkpoints/' + timestamp,
        'history_path': './history/' + timestamp,
        'results_path': './results/' + timestamp
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

    batch_size = 120

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
    #model = MultiHeadClassifier().to(device)
    model = ResNet_Ensemble_debugged().to(device)
    learning_rate = 0.001
    criterion = F1Loss()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion3 = nn.CrossEntropyLoss()
    criterion4 = nn.CrossEntropyLoss()
    criterion5 = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    plot_bound = 0

    ######## Prediction Generation ########
    print(f"Prediction | Prediction start @ {get_time()}", flush=True)

    submission = pd.read_csv(os.path.join(location['base_path'], 'eval/info.csv'))
    predictions = []

    # Load best model selected by validation loss
    print(f"Prediction | Loading model (epoch with best validation loss)")
    #if best_epoch != epoch + 1:
    #    checkpoint = torch.load(os.path.join(location['checkpoints_path'], f'epoch{best_epoch}.pt'), map_location=device)
    #    model.load_state_dict(checkpoint['model'])
    checkpoint = torch.load('./epoch14.pt', map_location=device)
    model.eval()
    with torch.no_grad():
        for batch_index, images in enumerate(test_loader):
            print('Prediction | Batch {} / {} start'.format(batch_index + 1, test_batches), flush=True)

            images = images.to(device)
            #ResNet_Ensemble1,2 : output1, output2, output3, output4, output5, outputs = model(images)
            _, _, _, _, _, _, outputs = model(images) #ResNet_Ensemble_debugged
            #ResNet_Ensemble1,2 : output = output1+output2+output3+output4+output5+outputs
            prediction = torch.argmax(outputs, dim=1)
            predictions.extend(prediction.cpu().numpy())
            
            #prediction = torch.argmax(outputs, dim=1)
            #predictions.extend(prediction.cpu().numpy())

    # Save predictions
    submission['ans'] = predictions
    submission.to_csv(os.path.join(location['results_path'], 'submission.csv'), index=False)

    print(f"Prediction | Finished prediction @ {get_time()}", flush=True)

    
if __name__ == '__main__':
    # Last checkpoint's training position
    done_epochs = 0

    # How much epochs to train now
    train_epochs = 15

    train_and_eval(done_epochs, train_epochs, clear_log=False)