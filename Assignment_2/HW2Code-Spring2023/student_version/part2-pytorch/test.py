# NOTE: The scaffolding code for this part of the assignment
# is adapted from https://github.com/pytorch/examples.
from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from challenge_data import ChallengeData
from models import MyModel

# Training settings
parser = argparse.ArgumentParser(description='CIFAR-10 Evaluation Script')
parser.add_argument('--model',
                    help='full path of model to evaluate')
parser.add_argument('--test-dir', default='data',
                    help='directory that contains test_images.npy file ')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA evaluation')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Load CIFAR10 using torch data paradigm
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
test_dataset = ChallengeData(args.test_dir, download=False,
                        transform=transform)
# Datasets
test_loader = torch.utils.data.DataLoader(test_dataset,
                 batch_size=args.test_batch_size, shuffle=False, **kwargs)

if os.path.exists(args.model):
    model = MyModel()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
else:
    print('Model path specified does not exst')
    sys.exit(1)

# cross-entropy loss function
criterion = F.cross_entropy
if args.cuda:
    model.cuda()


def evaluate():
    '''
    Compute loss on test data.
    '''
    model.eval()
    loader = test_loader
    predictions = []
    for batch_i, batch in enumerate(loader):
        data = batch
        if args.cuda:
            data= data.cuda()
        data = Variable(data, volatile=True)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        predictions += pred.reshape(-1).tolist()
        print('Batch:{}'.format(batch_i))
    return predictions

predictions = evaluate()

with open('predictions.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(['image_id', 'label'])
    for i, p in enumerate(predictions):
        csv_writer.writerow([i, int(p)])
