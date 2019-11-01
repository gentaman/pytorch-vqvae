import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from modules.modules import VectorQuantizedVAE, to_scalar
from modules.functions import Classifier

from utils import accuracy
from utils import copy_model
from datasets.datasets import MiniImagenet, get_dataset

from tensorboardX import SummaryWriter


def train(data_loader, model, clfy, optimizer, args, writer=None, loss_fn=None):
    for images, labels in tqdm(data_loader, total=len(data_loader)):
        # print(images.shape) 
        images = images.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x, z_q_x_st = model(images)
        # print(x_tilde.shape)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        preds = clfy(z_q_x_st)
        # predict label
        loss_pred = loss_fn(preds, labels)
        acc, = accuracy(preds, labels)

        loss = args.recon_coeff * loss_recons + args.vq_coeff * loss_vq + args.beta * loss_commit + args.gamma * loss_pred
        loss.backward()

        if writer is not None:
            # Logs
            writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
            writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)
            writer.add_scalar('loss/train/prediction', loss_pred.item(), args.steps)
            writer.add_scalar('accuracy/train', acc.item(), args.steps)

        optimizer.step()
        args.steps += 1

def test(data_loader, model, clfy, args, writer=None, loss_fn=None):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        loss_pred = 0.
        acc_total = 0.
        for images, labels in data_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            x_tilde, z_e_x, z_q_x, z_q_x_st = model(images)
            preds = clfy(z_q_x_st)
            acc, = accuracy(preds, labels)

            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)
            loss_pred += loss_fn(preds, labels)
            acc_total += acc

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)
        loss_pred /= len(data_loader)
        acc_total /= len(data_loader)
    
    if writer is not None:
        # Logs
        writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/test/quantization', loss_vq.item(), args.steps)
        writer.add_scalar('loss/test/prediction', loss_pred.item(), args.steps)
        writer.add_scalar('accuracy/test', acc_total.item(), args.steps)

    res = {
        'recons': loss_recons.item(),
        'vq': loss_vq.item(),
        'pred': loss_pred.item(),
        'acc': acc_total.item()
    }
    return res

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _, _ = model(images)
    return x_tilde

def main(args):
    root = args.root
    path = os.path.join(root, 'logs', args.output_folder)
    writer = SummaryWriter(path)
    save_filename = os.path.join(root, 'models', args.output_folder)
    
    result = get_dataset(args.dataset, args.data_folder, image_size=args.image_size)

    train_dataset = result['train']
    test_dataset = result['test']
    valid_dataset = result['valid']
    num_channels = result['num_channels']

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.batch_size, shuffle=True)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = VectorQuantizedVAE(
        num_channels, args.hidden_size, args.k, pred=True, transpose=args.resblock_transpose, BN=args.BN
        ).to(args.device)

    if args.hidden_fmap_size is None:
        hidden_fmap_size = args.image_size // 4
    else:
        hidden_fmap_size = args.hidden_fmap_size
    if args.gap:
        n_in = args.hidden_size
    else:
        n_in = int(args.hidden_size * hidden_fmap_size * hidden_fmap_size)
    n_out = len(train_dataset._label_encoder)
    predictor = Classifier(n_in, n_out, gap=args.gap).to(args.device)

    if args.model:
        print("load model ==> {}".format(args.model))
        with open(args.model, 'rb') as f:
            state_dict = torch.load(f)
            try:
                model = copy_model(state_dict, model, verbose=1)
            except:
                model.load_state_dict(state_dict)
    if args.predictor:
        print("load predictor ==> {}".format(args.model))
        with open(args.predictor, 'rb') as f:
            state_dict = torch.load(f)
            predictor.load_state_dict(state_dict)
    optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': predictor.parameters()}],
        lr=args.lr
        )

    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, args)
    grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('reconstruction', grid, 0)

    with open(os.path.join(save_filename, 'init.model.pt'), 'wb') as f:
        torch.save(model.state_dict(), f)
    with open(os.path.join(save_filename, 'init.predictor.pt'), 'wb') as f:
        torch.save(predictor.state_dict(), f)
    
    best_loss = -1.
    for epoch in tqdm(range(args.num_epochs), total=args.num_epochs):
        train(train_loader, model, predictor, optimizer, args, writer, predictor.loss)
        losses = test(valid_loader, model, predictor, args, writer, predictor.loss)
        loss = losses['recons']

        reconstruction = generate_samples(fixed_images, model, args)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        writer.add_image('reconstruction', grid, epoch + 1)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
            with open('{0}/best_predictor.pt'.format(save_filename), 'wb') as f:
                torch.save(predictor.state_dict(), f)
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(model.state_dict(), f)
        if args.gap:
            with open('{0}/predictor_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
                torch.save(predictor.state_dict(), f)
        
if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp
    import sys

    parser = argparse.ArgumentParser(description='VQ-VAE with prediction')

    # General
    parser.add_argument('--data-folder', type=str,
        help='name of the data folder')
    parser.add_argument('--dataset', type=str,
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')
    parser.add_argument('--image-size', type=int, default=128,
        help='size of the input image (default: 128)')
    parser.add_argument('--model', type=str, default='',
        help='filename containing the model')
    parser.add_argument('--predictor', type=str, default='',
        help='filename containing the predictor')
    parser.add_argument('--gap', action='store_true',
        help='add GAP')
    parser.add_argument('--off-bn', dest='BN', action='store_false',
        help='disable Batch Noramalization')
    parser.add_argument('--resblock-transpose', action='store_true',
        help='apply conv transpose to ResBlock')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')

    parser.add_argument('--hidden-fmap-size', type=int, default=None,
        help='size of the latent vectors (default: None)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=1.0,
        help='contribution of prediction loss, between 0.1 and 2.0 (default: 1.0)')
    parser.add_argument('--recon_coeff', type=float, default=1.0,
        help='contribution of reconstruction loss, between 0.1 and 2.0 (default: 1.0)')
    parser.add_argument('--vq_coeff', type=float, default=1.0,
        help='contribution of vector quantization loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--root', type=str, default='.',
        help='name of the root of the output folder (default: .)')
    parser.add_argument('--output-folder', type=str, default='vqvae',
        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda, default: cpu)')

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

