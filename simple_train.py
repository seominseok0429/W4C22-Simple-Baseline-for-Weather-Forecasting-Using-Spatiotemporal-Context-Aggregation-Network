import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.utils.data import random_split, DataLoader
from rainf_multi_all import RainFDataset
import numpy as np
from sianet import sianet
from utils import progress_bar
import os

modalities = ['radar', 'hima01', 'hima06', 'imerg', 'rain', 'temp', 'hum', 'wsp', 'wdir']

train_dataset = RainFDataset(data_path = '/workspace/DATA/npy/', years = [2017, 2018], input_seq=6,
        img_size=256, modalities=modalities, cls_num = 3, normalize_hsr = True, use_meta=False)
val_dataset = RainFDataset(data_path = '/workspace/DATA/npy/', years = [2019], input_seq=6,
        img_size=256, modalities=modalities, cls_num = 3, normalize_hsr = True, use_meta=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

model = sianet()
model = model.cuda()
model = torch.nn.DataParallel(model)
cudnn.benchmark = True

optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4, weight_decay=0.1)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0

    for batch_idx, (inputs, targets, _, _, _) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.long().cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
         train_loss += loss.item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f |' % (train_loss/(batch_idx+1)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _, _, _) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.long().cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f |'
                         % (test_loss/(batch_idx+1)))

        state = {
            'net': model.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')

for epoch in range(0, 100):
    train(epoch)
    test(epoch)
    scheduler.step()
