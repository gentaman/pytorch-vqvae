import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from modules.modules import AE, to_scalar
from modules.functions import Classifier
from datasets.datasets import MiniImagenet, get_dataset

from utils import accuracy
from utils import copy_model
from tensorboardX import SummaryWriter

def train(data_loader, model, clfy, optimizer, args, writer=None, loss_fn=None):
    assert isinstance(model, AE)

    for images, lables in tqdm(data_loader, total=len(data_loader)):
        # print(images.shape)
        images = images.to(args.device)
        labels = lables.to(args.device)

        optimizer.zero_grad()
        x_tilde, code = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)

        if args.gap:
            preds = clfy(code)
        else:
            preds = clfy(code.view(code.size(0), -1))
        # print(sample.size())

        loss_pred = loss_fn(preds, labels)
        acc, = accuracy(preds, labels)

        loss = args.recon_coeff * loss_recons + args.gamma * loss_pred
        loss.backward()

        if writer is not None:
            # Logs
            writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
            writer.add_scalar('loss/train/prediction', loss_pred.item(), args.steps)
            writer.add_scalar('accuracy/train', acc.item(), args.steps)

        optimizer.step()
        args.steps += 1

def test(data_loader, model, clfy, args, writer=None, loss_fn=None):
    assert isinstance(model, AE)

    with torch.no_grad():
        loss_recons = 0.
        loss_pred = 0.
        acc_total = 0.
        for images, labels in data_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            x_tilde, sample = model(images)
            if args.gap:
                preds = clfy(sample)
            else:
                preds = clfy(sample.view(sample.size(0), -1))

            acc, = accuracy(preds, labels)

            loss_recons += F.mse_loss(x_tilde, images)
            loss_pred += loss_fn(preds, labels)
            acc_total += acc

        loss_recons /= len(data_loader)
        loss_pred /= len(data_loader)
        acc_total /= len(data_loader)
    
    if writer is not None:
        # Logs
        writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/test/prediction', loss_pred.item(), args.steps)
        writer.add_scalar('accuracy/test', acc_total.item(), args.steps)

    res = {
        'recons': loss_recons.item(),
        'pred': loss_pred.item(),
        'acc': acc_total.item()
    }

    return res

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _ = model(images)
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
        batch_size=16, shuffle=True)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = AE(num_channels, args.hidden_size, pred=True).to(args.device)
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
    from utils import get_args
    import os
    import sys

    args = get_args(description='AE with prediction')

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
