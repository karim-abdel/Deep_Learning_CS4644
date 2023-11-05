"""
Helper functions.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import math
import time
import random

# Pytorch packages
import torch
import torch.optim as optim
import torch.nn as nn

# Numpy
import numpy as np

# Tqdm progress bar
from tqdm import tqdm_notebook

RANDOM_SEED = 0


def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def set_seed_nb():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED + 1)


def deterministic_init(net: nn.Module):
    for p in net.parameters():
        if p.data.ndimension() >= 2:
            set_seed_nb()
            nn.init.xavier_uniform_(p.data)
        else:
            nn.init.zeros_(p.data)


def train(model, dataloader, optimizer, criterion, scheduler=None, device='cpu'):
    model.train()

    # Record total loss
    total_loss = 0.

    # Get the progress bar for later modification
    progress_bar = tqdm_notebook(dataloader, ascii=True)

    # Mini-batch training
    for batch_idx, data in enumerate(progress_bar):
        source = data[0].transpose(1, 0).to(device)
        target = data[1].transpose(1, 0).to(device)

        translation = model(source)
        translation = translation.reshape(-1, translation.shape[-1])
        target = target.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(translation, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_description_str(
            "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device='cpu'):
    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        # Get the progress bar
        progress_bar = tqdm_notebook(dataloader, ascii=True)
        for batch_idx, data in enumerate(progress_bar):
            source = data[0].transpose(1, 0).to(device)
            target = data[1].transpose(1, 0).to(device)

            translation = model(source)
            translation = translation.reshape(-1, translation.shape[-1])
            target = target.reshape(-1)

            loss = criterion(translation, target)
            total_loss += loss.item()
            progress_bar.set_description_str(
                "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss


def unit_test_values(testcase):
    if testcase == 'rnn':
        return torch.FloatTensor([[-0.9080, -0.5639, -3.5862],
                                  [-1.2683, -0.4294, -2.6910],
                                  [-1.7300, -0.3964, -1.8972],
                                  [-2.3217, -0.4933, -1.2334]]), torch.FloatTensor([[0.9629,  0.9805, -0.5052,  0.8956],
                                                                                    [0.7796,  0.9508, -
                                                                                        0.2961,  0.6516],
                                                                                    [0.1039,  0.8786, -
                                                                                        0.0543,  0.1066],
                                                                                    [-0.6836,  0.7156,  0.1941, -0.5110]])

    if testcase == 'lstm':
        ht = torch.FloatTensor([[-0.0452,  0.7843, -0.0061,  0.0965],
                                [-0.0206,  0.5646, -0.0246,  0.7761],
                                [-0.0116,  0.3177, -0.0452,  0.9305],
                                [-0.0077,  0.1003,  0.2622,  0.9760]])
        ct = torch.FloatTensor([[-0.2033,  1.2566, -0.0807,  0.1649],
                                [-0.1563,  0.8707, -0.1521,  1.7421],
                                [-0.1158,  0.5195, -0.1344,  2.6109],
                                [-0.0922,  0.1944,  0.4836,  2.8909]])
        return ht, ct

    if testcase == 'encoder':
        expected_out = torch.FloatTensor([[[-0.7773, -0.2031],
                                         [-0.6186, -0.2321]],

                                        [[ 0.0599, -0.0151],
                                         [-0.9237,  0.2675]],

                                        [[ 0.6161,  0.5412],
                                         [ 0.7036,  0.1150]],

                                        [[ 0.6161,  0.5412],
                                         [-0.5587,  0.7384]],

                                        [[-0.9062,  0.2514],
                                         [-0.8684,  0.7312]]])
        expected_hidden = torch.FloatTensor([[[ 0.4912, -0.6078],
                                         [ 0.4932, -0.6244],
                                         [ 0.5109, -0.7493],
                                         [ 0.5116, -0.7534],
                                         [ 0.5072, -0.7265]]])
        return expected_out, expected_hidden

    if testcase == 'decoder':
        expected_out = torch.FloatTensor(
        [[-2.1507, -1.6473, -3.1772, -3.2119, -2.6847, -2.1598, -1.9192, -1.8130,
         -2.6142, -3.1621],
        [-1.9727, -2.1730, -3.3104, -3.1552, -2.4158, -1.7287, -2.1686, -1.7175,
         -2.6946, -3.2259],
        [-2.1952, -1.7092, -3.1261, -2.9943, -2.5070, -2.1580, -1.9062, -1.9384,
         -2.4951, -3.1813],
        [-2.1961, -1.7070, -3.1257, -2.9950, -2.5085, -2.1600, -1.9053, -1.9388,
         -2.4950, -3.1811],
        [-2.7090, -1.1256, -3.0272, -2.9924, -2.8914, -3.0171, -1.6696, -2.4206,
         -2.3964, -3.2794]])
        expected_hidden = torch.FloatTensor([[
                                            [-0.1854,  0.5561],
                                            [-0.6016,  0.0276],
                                            [ 0.0255,  0.3106],
                                            [ 0.0270,  0.3131],
                                            [ 0.9470,  0.8482]]])
        return expected_out, expected_hidden

    if testcase == 'seq2seq':
        expected_out = torch.FloatTensor(
        [[[-2.4136, -2.2861, -1.7145, -2.5612, -1.9864, -2.0557, -1.7461,
          -2.1898],
         [-2.0869, -2.9425, -2.0188, -1.6864, -2.5141, -2.3069, -1.4921,
          -2.3045]],

        [[-1.8506, -2.3783, -2.1297, -1.9083, -2.5922, -2.3552, -1.5708,
          -2.2505],
         [-2.0939, -2.1570, -2.0352, -2.2691, -2.1251, -1.8906, -1.8156,
          -2.3654]]]
        )
        return expected_out

    if testcase == 'attention':

        hidden = torch.FloatTensor(
            [[[-0.7232, -0.6048],
              [0.9299, 0.7423],
              [-0.4391, -0.7967],
              [-0.0012, -0.2803],
              [-0.3248, -0.3771]]]
        )

        enc_out = torch.FloatTensor(
            [[[-0.7773, -0.2031],
              [-0.6186, -0.2321]],

             [[0.0599, -0.0151],
              [-0.9237, 0.2675]],

             [[0.6161, 0.5412],
              [0.7036, 0.1150]],

             [[0.6161, 0.5412],
              [-0.5587, 0.7384]],

             [[-0.9062, 0.2514],
              [-0.8684, 0.7312]]]
        )

        expected_attention = torch.FloatTensor(
            [[[0.4902, 0.5098]],

             [[0.7654, 0.2346]],

             [[0.4199, 0.5801]],

             [[0.5329, 0.4671]],

             [[0.6023, 0.3977]]]
        )
        return hidden, enc_out, expected_attention

    if testcase == 'seq2seq_attention':
        expected_out = torch.FloatTensor(
            [[[-2.8071, -2.4324, -1.7512, -2.7194, -1.7530, -2.1202, -1.6578,
               -2.0519],
              [-2.2137, -2.4308, -2.0972, -2.1079, -1.9882, -2.0411, -1.6965,
               -2.2229]],

             [[-1.9549, -2.4265, -2.1293, -1.9744, -2.2882, -2.4210, -1.4311,
               -2.4892],
              [-2.1284, -2.2369, -2.1940, -1.9027, -2.1065, -2.2274, -1.7391,
               -2.2220]]]
        )
        return expected_out

