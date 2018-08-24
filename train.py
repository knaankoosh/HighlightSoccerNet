import time
import os
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data import initialize_loaders
from main import setup_main, to_variables, ModelSaver, update_stats
from models import crNN

def load_or_init_models(model, opt):
    if opt.net != '':
        model.load_state_dict(torch.load(opt.net))
    return model

def train(net, criterion, optimizer, data, opt, lambda_pixel=100):
    net.train()
    optimizer.zero_grad()

    frames, volumes, labels = data
    batch_size, seq_len, H, W, C = frames.size()

    losses = torch.zeros(batch_size)
    for i in range(batch_size):
        print(i)
        frame = frames[i]
        audio = volumes[i]
        output = net(frame, audio)
        losses[i] = criterion(output, labels[i])

    losses.backward()
    optimizer.step()

    return losses

def test(net, criterion, data, opt, lambda_pixel=100):
    net.eval()
    loss = 0

    return loss


def run(opt):
    train_loader, test_loader = initialize_loaders(opt)

    # Initialize net
    net = load_or_init_models(crNN.crNN(130), opt)

    # Optimizers
    # optimizer = torch.optim.Adam(net.cnn_params, lr=opt.lr, betas=(opt.b1, opt.b2))
    # optimizer.add_param_group({'rnn_params': net.rnn_params})
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    net, criterion = to_variables((net, torch.nn.BCELoss()), cuda=opt.cuda, device=opt.device)

    net_saver = ModelSaver(f'{opt.checkpoint_dir}/saved_models/{opt.name}')

    train_writer = SummaryWriter(log_dir=os.path.join(opt.checkpoint_dir, 'train'))
    test_writer = SummaryWriter(log_dir=os.path.join(opt.checkpoint_dir, 'test'))
    prev_time = time.time()

    for epoch in tqdm(range(opt.epoch, opt.n_epochs), desc='Training'):
        # Training
        avg_stats = defaultdict(float)
        for i, data in enumerate(train_loader):
            data = to_variables(data, cuda=opt.cuda, device=opt.device)
            loss = train(net, criterion, optimizer, data, opt)
            update_stats(avg_stats, loss)

        # Log training progress
        str_out = '[train] {}/{} '.format(epoch, opt.n_epochs)
        for k, v in avg_stats.items():
            avg = v / len(train_loader)
            train_writer.add_scalar(k, avg, epoch)
            str_out += '{}: {:.6f}  '.format(k, avg)
        print(str_out)

        # Testing
        avg_stats = defaultdict(float)
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = to_variables(data, cuda=opt.cuda, device=opt.device, test=True)
                losses = test(net, criterion, data, opt)
                update_stats(avg_stats, losses)

        # Log testing progress
        str_out = '[test] {}/{} '.format(epoch, opt.n_epochs)
        for k, v in avg_stats.items():
            avg = v / len(test_loader)
            test_writer.add_scalar(k, avg, epoch)
            str_out += '{}: {:.6f}  '.format(k, avg)
        print(str_out)

        if epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            net_saver.save_if_best(net, losses)


if __name__ == '__main__':
    opt = setup_main()
    run(opt)
