import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import tuh_dataset
from vit_pytorch.vit import ViT
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def thresh_max_f1(y_true, y_prob):
    """
    Find best threshold based on precision-recall curve to maximize F1-score.
    Binary calssification only
    """
    if len(set(y_true)) > 2:
        raise NotImplementedError

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresh_filt = []
    fscore = []
    n_thresh = len(thresholds)
    for idx in range(n_thresh):
        curr_f1 = (2 * precision[idx] * recall[idx]) / \
            (precision[idx] + recall[idx])
        if not (np.isnan(curr_f1)):
            fscore.append(curr_f1)
            thresh_filt.append(thresholds[idx])
    # locate the index of the largest f score
    ix = np.argmax(np.array(fscore))
    best_thresh = thresh_filt[ix]
    return best_thresh


def seed_everything(seed=99):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()

device = 'cuda'
model = ViT(image_size=(3200, 15), patch_size=(64, 5), num_classes=1, dim=16, depth=4, heads=4, mlp_dim=4, pool='cls',
            channels=1, dim_head=4, dropout=0.2, emb_dropout=0.2).to(device)
sigmoid = nn.Sigmoid()

# Training settings
batch_size = 257
epochs = 40
lr = 3e-5
gamma = 0.7
tuh_dataset.args.eeg_type = 'stft'

# loss function
criterion = nn.BCELoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
print('patient', tuh_dataset.args.patient_number)
train_loader, val_loader, test_loader = tuh_dataset.get_data_loader(batch_size,
                                                                    save_dir=tuh_dataset.args.save_directory)

best_val_auc = 0.0
model_directory = os.path.join(tuh_dataset.args.save_directory, 'Siena_model{}'.format(tuh_dataset.args.patient_number))
os.mkdir(model_directory)
for epoch in range(epochs):

    model.train()
    train_label_all = []
    train_prob_all = []
    epoch_train_loss = 0
    for data, label in tqdm(train_loader, desc='Training '):
        train_label_all.extend(label)

        data = data.to(device)
        label = label.to(device).float()

        prob = model(data)
        prob = torch.squeeze(sigmoid(prob))
        train_prob_all.extend(prob.cpu().detach().numpy())

        loss = criterion(prob, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss / len(train_loader)

    train_auc = roc_auc_score(train_label_all, train_prob_all)

    model.eval()
    val_label_all = []
    val_prob_all = []
    epoch_val_loss = 0
    with torch.no_grad():
        for data, label in tqdm(val_loader, desc='Evaluation '):
            val_label_all.extend(label.cpu().numpy())

            data = data.to(device)
            label = label.to(device).float()

            val_prob = model(data)
            val_prob = torch.squeeze(sigmoid(val_prob))
            val_prob_all.extend(val_prob.cpu().numpy())

            val_loss = criterion(val_prob, label)

            epoch_val_loss += val_loss / len(val_loader)

    val_auc = roc_auc_score(val_label_all, val_prob_all)
    best_th = thresh_max_f1(val_label_all, val_prob_all)
    print(best_th)
    val_predict = np.where(val_prob_all > best_th, 1, 0)
    val_f1 = f1_score(val_label_all, val_predict)

    print(f"Epoch: {epoch + 1} - train_loss: {epoch_train_loss:.4f} -  train_auc: {train_auc:.4f}; \n"
          f"val_loss: {epoch_val_loss:.4f}\n - val_auc: {val_auc:.4f} - val_f1: {val_f1:.4f}")

    if best_val_auc < val_auc:
        best_val_auc = val_auc

        model.eval()
        test_label_all = []
        test_prob_all = []
        epoch_test_loss = 0
        with torch.no_grad():
            for data, label in tqdm(test_loader, desc='Testing '):
                test_label_all.extend(label)

                data = data.to(device)
                label = label.to(device).float()

                test_prob = model(data)
                test_prob = torch.squeeze(sigmoid(test_prob))
                test_prob_all.extend(test_prob.cpu().numpy())

                test_loss = criterion(test_prob, label)

                epoch_test_loss += test_loss / len(test_loader)

        test_auc = roc_auc_score(test_label_all, test_prob_all)
        test_predict = np.where(test_prob_all > best_th, 1, 0)
        test_f1 = f1_score(test_label_all, test_predict)
        print(f"test_loss: {epoch_test_loss:.4f} - test_auc: {test_auc:.4f}")
        print(f"test_f1: {test_f1:.4f}")

        torch.save(model, os.path.join(model_directory, 'test_model_{}_{}'.format(epoch, test_auc)))
