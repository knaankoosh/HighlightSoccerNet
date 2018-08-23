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
import models


def trainG(generator, criterion_GAN, criterion_pixelwise, optimizer, data, opt, lambda_pixel=100):
    generator.train()
    optimizer.zero_grad()

    x_hr, x_lr, y_hr, y_lr = data

    # GAN loss
    y_hat = generator(x_hr, x_lr)

    # Pixel-wise loss
    loss_pixel = criterion_pixelwise(y_hat, y_hr)

    # Total loss
    loss_G = lambda_pixel * loss_pixel

    loss_G.backward()
    optimizer.step()

    return y_hat, {'loss_G': loss_G, 'loss_pixel': loss_pixel}

def test(generator, criterion_pixelwise, data, opt, lambda_pixel=100):
    generator.eval()
    x_hr, x_lr, y_hr, y_lr = data


    # GAN loss
    y_hat = generator(x_hr, x_lr)

    # Loss from discriminator

    # Pixel-wise loss
    loss_pixel = criterion_pixelwise(y_hat, y_hr)

    # Total loss
    loss_G = lambda_pixel * loss_pixel

    return y_hat, {'loss_G': loss_G, 'loss_pixel': loss_pixel}


def run(opt):
    train_loader, test_loader = initialize_loaders(opt)

    # Initialize generator and discriminator
    cNN = load_or_init_models(RetouchGenerator(opt.device), opt)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    generator, criterion_GAN, criterion_pixelwise = to_variables((generator,
                                                                  torch.nn.BCEWithLogitsLoss(),
                                                                  torch.nn.MSELoss()),
                                                                  cuda=opt.cuda,
                                                                  device=opt.device)

    saverG = ModelSaver(f'{opt.checkpoint_dir}/saved_models/{opt.name}')
    saverD = ModelSaver(f'{opt.checkpoint_dir}/saved_models/{opt.name}')
    train_writer = SummaryWriter(log_dir=os.path.join(opt.checkpoint_dir, 'train'))
    test_writer = SummaryWriter(log_dir=os.path.join(opt.checkpoint_dir, 'test'))
    prev_time = time.time()

    for epoch in tqdm(range(opt.epoch, opt.n_epochs), desc='Training'):

        ####
        # Train
        ###
        avg_stats = defaultdict(float)
        for i, data in enumerate(train_loader):
            data = to_variables(data, cuda=opt.cuda, device=opt.device)
            y_hat, loss_G = trainG(generator, criterion_GAN, criterion_pixelwise, optimizer_G, data, opt)
            update_stats(avg_stats, loss_G)

            # Print image to tensorboard
            if (epoch % opt.sample_interval == 0) and (i % 50 == 0):
                train_writer.add_image('RetouchNet', y_hat[0], epoch)
                train_writer.add_image('GroundTruth', data[0][0], epoch)
                train_writer.add_image('raw', data[2][0], epoch)


    # Log Progress
        str_out = '[train] {}/{} '.format(epoch, opt.n_epochs)
        for k, v in avg_stats.items():
            avg = v / len(train_loader)
            train_writer.add_scalar(k, avg, epoch)
            str_out += '{}: {:.6f}  '.format(k, avg)
        print(str_out)

        ####
        # Test
        ###
        avg_stats = defaultdict(float)
        images = None
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = to_variables(data, cuda=opt.cuda, device=opt.device, test=True)
                images, losses = test(generator, criterion_pixelwise, data, opt)
                update_stats(avg_stats, losses)

                # Print image to tensorboard
                if (epoch % opt.sample_interval == 0) and (i % 5 == 0):
                    test_writer.add_image('RetouchNet', images[0], epoch)
                    test_writer.add_image('GroundTruth', data[0][0], epoch)
                    test_writer.add_image('raw', data[2][0], epoch)

        # Log Progress
        str_out = '[test] {}/{} '.format(epoch, opt.n_epochs)
        for k, v in avg_stats.items():
            avg = v / len(test_loader)
            test_writer.add_scalar(k, avg, epoch)
            str_out += '{}: {:.6f}  '.format(k, avg)
        print(str_out)

        if epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            saverG.save_if_best(generator, loss_G['loss_G'])


if __name__ == '__main__':
    opt = setup_main()
    run(opt)
