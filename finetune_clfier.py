import json

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from tqdm import tqdm as tqdm
from torch import nn
from tensorboardX import SummaryWriter

from utils import accuracy
from datasets.datasets import get_dataset
from modules.modules import VectorQuantizedVAE

class Classifier(nn.Module):
    def __init__(self, in_f, out_f):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_f, out_f)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(data_loader, model, clfy, optimizer, args, writer=None, loss_fn=None):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    for images, labels in tqdm(data_loader, total=len(data_loader)):
        images = images.to('cuda')
        labels = labels.to('cuda')

        optimizer.zero_grad()
        with torch.no_grad():
            latents = model.encode(images)
            latents = model.codebook.embedding(latents).permute(0, 3, 1, 2)
        out = clfy(latents)
        loss = loss_fn(out, labels)
        loss.backward()

        if writer is not None:
            # Logs
            writer.add_scalar('loss/train', loss.item(), args.steps)

        optimizer.step()
        args.steps += 1
    
def test(data_loader, model, clfy, args, writer=None):
    with torch.no_grad():
        loss_total = 0.
        acc_total = 0.
        for images, labels in tqdm(data_loader, total=len(data_loader)):
            # print(images.shape)
            images = images.to('cuda')
            labels = labels.to('cuda')

            latents = model.encode(images)
            latents = model.codebook.embedding(latents).permute(0, 3, 1, 2)
            out = clfy(latents)
            loss_total += loss_fn(out, labels)
            acc, = accuracy(out, labels)
            acc_total += acc
            if writer is not None:
                # Logs
                writer.add_scalar('loss/test', loss.item(), args.steps)

        loss_total /= len(data_loader)
        acc_total /= len(data_loader)
            
    return loss_total.item(), acc_total.item()


def main(args):
    root = args.root
    path = os.path.join(root, 'logs', args.output_folder)
    writer = SummaryWriter(path)
    save_path = os.path.join(root, 'models', args.output_folder)

    result = get_dataset(args.dataset, args.data_folder, image_size=args.image_size)

    train_dataset = result['train']
    test_dataset = result['test']
    valid_dataset = result['valid']
    num_channels = result['num_channels']

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=16, shuffle=True)

    # Save the label encoder
    with open('{0}/labels.json'.format(save_path), 'w') as f:
        json.dump(train_dataset._label_encoder, f)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = VectorQuantizedVAE(num_channels, args.hidden_size_vae, args.k).to(args.device)    
    model_path = os.path.join(root, 'models', args.model)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    if args.multi_gpu:
        print('using multi-gpu')
        model = nn.DataParallel(model)
        cudnn.benchmark = True
    model.eval()

    if args.hidden_size is None:
        hidden_size = args.image_size // 4
    else:
        hidden_size = args.hidden_size

    n_out = len(train_dataset._label_encoder)
    predictor = Classifier(int(args.k*hidden_size*hidden_size), n_out).to(args.device)
    if args.multi_gpu:
        print('using multi-gpu')
        prior = nn.DataParallel(predictor)
        cudnn.benchmark = True
        
    optimizer = torch.optim.Adam(prior.parameters(), lr=args.lr)

    best_loss = -1.
    loss_fn = nn.CrossEntropyLoss()
    for epoch in tqdm(range(args.num_epochs), total=args.num_epochs):
        train(train_loader, model, predictor, optimizer, args, writer, loss_fn=loss_fn)
        # The validation loss is not properly computed since
        # the classes in the train and valid splits of Mini-Imagenet
        # do not overlap.
        # loss = test(valid_loader, model, prior, args, writer)
        
        # if (epoch == 0) or (loss < best_loss):
        #     best_loss = loss
        #     with open(os.path.join(save_path, 'best.pixelcnn.model'), 'wb') as f:
        #         torch.save(prior.state_dict(), f)
    with open(os.path.join(save_path, 'final.model'), 'wb') as f:
        torch.save(predictor.state_dict, f)

if __name__ == '__main__':
    import argparse
    import os
    import sys
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Finetune VQ-VAE latents')

    # General
    parser.add_argument('--data-folder', type=str,
        help='name of the data folder')
    parser.add_argument('--dataset', type=str,
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')
    parser.add_argument('--image-size', type=int, default=128,
        help='size of the input image (default: 128)')
    parser.add_argument('--model', type=str, required=True,
        help='filename containing the model')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=None,
        help='size of the latent vecotor (default: None)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-4,
        help='learning rate for Adam optimizer (default: 3e-4)')

    # Miscellaneous
    parser.add_argument('--root', type=str, default='.',
        help='name of the root of the output folder (default: .)')
    parser.add_argument('--output-folder', type=str, default='prior',
        help='name of the output folder (default: prior)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda, default: cpu)')
    parser.add_argument('--multi-gpu', action='store_true',
                        help='enable mutli-gpu.')

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    root = args.root
    log_path = os.path.join(root, 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    models_path = os.path.join(root, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    path = os.path.join(root, 'configs', args.output_folder)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'config'), 'w') as f:
        f.write('\n'.join(sys.argv))
    
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    path = os.path.join(models_path, args.output_folder)
    if not os.path.exists(path):
        os.makedirs(path)
    args.steps = 0

    main(args)