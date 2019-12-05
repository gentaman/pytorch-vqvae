import os
import pickle

from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SequentialSampler

import sys
sys.path.append('..')
from modules.modules import VectorQuantizedVAE
from modules.functions import Classifier
from datasets.datasets import get_dataset
from utils import accuracy
from utils.get_argument import get_args, parse_config
from datasets.cross_validation import kfold_cv, get_splited_dataloader
del sys.path[-1]


def get_receptive_field(neuron_index, layer_info, pad=(0, 0)):
    n, j, rf, start = layer_info
    if isinstance(neuron_index, tuple):
        center_y = start + (neuron_index[1]) * (j)
        center_x = start + (neuron_index[0]) * (j)
    else:
        center_y = start + (neuron_index // n) * (j)
        center_x = start + (neuron_index % n) * (j)
    return (center_x, center_y), (rf / 2, rf / 2)


def clip(x, min_val, max_val):
    x[x < min_val] = min_val
    x[max_val < x] = max_val
    return x

def get_rf_ragion(center, rf, n_in=32, clip=True):
    k = 0
    x1 = torch.floor(center[k].float() - rf[k])
    x2 = torch.floor(center[k].float() + rf[k])
    if clip:
        xs = (clip(x1, 0, n_in), clip(x2, 0, n_in))
    else:
        xs = (x1.long(), x2.long())
    k = 1
    x1 = torch.floor(center[k].float() - rf[k])
    x2 = torch.floor(center[k].float() + rf[k])
    if clip:
        ys = (clip(x1, 0, n_in), clip(x2, 0, n_in))
    else:
        ys = (x1.long(), x2.long())
    return xs, ys

def plot_rf_imgs(best_loss_valids, ks=[16, 32, 64, 128, 256, 512], out_dir='.',
                 num_channels=1, test_loader=None, final_rf=None, rf=None, args=None):
    def test(a, b, c, d):
        _index1 = torch.arange(a, b)
        _index2 = torch.arange(c, d)
        X, Y = torch.meshgrid(_index1, _index2)
        return X, Y
    device = args.device
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for n_cv in range(len(best_loss_valids)):
        model_dir_path = best_loss_valids[n_cv][2]
        basename = os.path.basename(model_dir_path)
        out_image_dir = os.path.join(out_dir, basename)
        if not os.path.exists(out_image_dir):
            os.makedirs(out_image_dir)
        hidden_size = args.hidden_size
        k_s = ks[n_cv]
        model = VectorQuantizedVAE(num_channels, hidden_size, k_s, pred=True, transpose=args.resblock_transpose).to(device)
        n_in = hidden_size
        n_out = 10
        predictor = Classifier(n_in, n_out, gap=args.gap).to(device)
        
        model_path = os.path.join(model_dir_path, 'best.pt')
        print('load model ==> {}'.format(model_path))
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))

        model_path = os.path.join(model_dir_path, 'best_predictor.pt')
        print('load model ==> {}'.format(model_path))
        with open(model_path, 'rb') as f:
            predictor.load_state_dict(torch.load(f))

        mean_imgs_k = torch.zeros(1, int(2 * rf[0]), int(2 * rf[1]), dtype=torch.float).to(device)
        mean_imgs = [list() for _ in range(k_s)]
        _cnt = 0
        cnts_k = [0 for _ in range(k_s)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            latents = model.encode(images)
            batch_size, fmap_size, fmap_size = latents.shape
            for k in range(k_s):
                latent_k_index = (latents == k)
                tmp = torch.arange(int(batch_size * fmap_size * fmap_size)).reshape(batch_size, fmap_size, fmap_size).long().to(device)
                act_neuron_index = tmp[latent_k_index]

                b_index = act_neuron_index // (fmap_size**2)
                n_index = act_neuron_index % (fmap_size**2)
                center, rf = get_receptive_field(n_index, final_rf)
                pad_size = 16
                xs, ys = get_rf_ragion(center, rf, clip=False)
                _index1 = ys[0] + pad_size
                _index2 = ys[1] + pad_size
                _index3 = xs[0] + pad_size
                _index4 = xs[1] + pad_size
                padded_images = F.pad(images, (pad_size, pad_size, pad_size, pad_size), 'constant', 0.)
                if len(b_index) == 0:
                    continue
                bs = torch.repeat_interleave(b_index, int(2 * rf[0] * 2 * rf[1]))
                _indeces = torch.stack([
                    torch.stack(test(i1, i2, i3, i4)) for i1, i2, i3, i4 in zip(_index1, _index2, _index3, _index4)
                ], dim=1).reshape(2, -1)
                tmp = padded_images[bs, :, _indeces[0], _indeces[1]]
                tmp = tmp.reshape(len(b_index), 1, int(2 * rf[0]), int(2 * rf[1]))
                mean_imgs[k].append(tmp)
                cnts_k[k] += len(b_index)

        for k in range(k_s):
            if len(mean_imgs[k]) > 0:
                mean_imgs[k] = torch.cat(mean_imgs[k], dim=0).sum(0)
        
        m = int(np.ceil(np.sqrt(np.sum(np.array(cnts_k) != 0))))
        plt.figure(figsize=(m * 2, m * 2))
        _i = 0
        for k in range(k_s):
            if cnts_k[k] == 0:
                continue
            else:
                tmp = mean_imgs[k].detach().cpu().numpy()[0] / cnts_k[k]
            plt.subplot(m, m, _i + 1)
            plt.title('codebook {}, N {}'.format(k, cnts_k[k]))
            plt.axis('off')
            plt.imshow(tmp, vmin=-1, vmax=1)
            _i += 1
        plt.tight_layout()
        path = os.path.join(out_image_dir, 'image_rec.png')
        plt.savefig(path)
        plt.close()

def main(main_args):
    dnames = [
    'mnist_im32_gap_k16_e100',
    'mnist_im32_gap_k32_e100',
    'mnist_im32_gap_k64_e100',
    'mnist_im32_gap_k128_e100',
    'mnist_im32_gap_k256_e100',
    'mnist_im32_gap_k512_e100'
    ]
    best_loss_valids = []

    if main_args.best:
        print('load == > {}'.format(main_args.best))
        with open(main_args.best, 'rb') as f:
            best_loss_valids = pickle.load(f)
        print(best_loss_valids)

    init = True
    end = False
    for configs in main_args.configs:
        configs_path = configs
        for dname in dnames:
            if end:
                break
            print(dname)
            config_path = os.path.join(configs_path, dname, 'config')
            args = get_args(description='VQ-VAE', inputs=['--config', config_path])

            if init:
                image_size = args.image_size
                dataset = args.dataset
                data_folder = args.data_folder
                result = get_dataset(dataset, data_folder, image_size=image_size)

                train_dataset = result['train']
                valid_dataset = result['valid']
                test_dataset = result['test']
                num_channels = result['num_channels']
                targets = train_dataset.targets.numpy()
                train_data_index, test_data_index = kfold_cv(targets, kfold=args.kfold)
                train_loaders = []
                valid_loaders = []
                for index1, index2 in zip(train_data_index, test_data_index):
                    train_sampler = SequentialSampler(index1)
                    test_sampler = SequentialSampler(index2)

                    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
                        num_workers=args.num_workers, pin_memory=True)
                    valid_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=False, sampler=test_sampler,
                        num_workers=args.num_workers, pin_memory=True)

                    train_loaders.append(train_loader)
                    valid_loaders.append(valid_loader)

                test_loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
                if len(best_loss_valids) > 0:
                    end = True
                init = False
            if end:
                break

            device = args.device
            image_size = args.image_size
            dataset = args.dataset
            data_folder = args.data_folder
            result = get_dataset(dataset, data_folder, image_size=image_size)

            num_channels = result['num_channels']
            hidden_size = args.hidden_size
            k = args.k
            model = VectorQuantizedVAE(num_channels, hidden_size, k, pred=True, transpose=args.resblock_transpose).to(args.device)
            n_in = hidden_size
            n_out = 10
            predictor = Classifier(n_in, n_out, gap=args.gap).to(device)

            best_loss_valid = [-1, None, None]
            for n_cv in tqdm(range(args.kfold), total=args.kfold):
                m_name = 'models_{}'.format(n_cv)
                model_path = os.path.join(args.root, m_name, args.output_folder, 'best.pt')
                print('load model ==> {}'.format(model_path))
                with open(model_path, 'rb') as f:
                    model.load_state_dict(torch.load(f))

                model_path = os.path.join(args.root, m_name, args.output_folder, 'best_predictor.pt')
                print('load model ==> {}'.format(model_path))
                with open(model_path, 'rb') as f:
                    predictor.load_state_dict(torch.load(f))

                model.eval()
                losses = []
                for images, labels in valid_loaders[n_cv]:
                    images = images.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        x_tilde, _, z_q_x, _ = model(images)
                        preds = predictor(z_q_x)

                        loss_recon = F.mse_loss(x_tilde, images)
                        loss_pred = F.cross_entropy(preds, labels)

                        loss = args.recon_coeff * loss_recon + args.gamma * loss_pred

                    losses.append(loss)
                value = torch.mean(torch.stack(losses))
                if best_loss_valid[0] == -1:
                    best_loss_valid[0] = os.path.join(args.root, m_name, args.output_folder)
                    best_loss_valid[1] = value.item()
                    best_loss_valid[2] = os.path.dirname(model_path)
                elif value < best_loss_valid[1]:
                    best_loss_valid[0] = os.path.join(args.root, m_name, args.output_folder)
                    best_loss_valid[1] = value.item()
                    best_loss_valid[2] = os.path.dirname(model_path)    
            best_loss_valids.append(best_loss_valid)

    out_dir = os.path.join(os.path.dirname(configs_path), main_args.output_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, 'best_info.pkl')
    print('write best loss model == > {}'.format(out_path))
    with open(out_path, 'wb') as f:
        pickle.dump(best_loss_valids, f)
    
    print('write == > {}'.format(out_dir))
    # fontsize = 24
    # plt.rcParams["font.size"] = fontsize
    plot_rf_imgs(best_loss_valids, out_dir=out_dir, test_loader=test_loader, final_rf=(8, 4, 26, 2.0), rf=(13, 13), args=args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot codebook effect')
    parser.add_argument('-c', '--configs', type=str, nargs='+', required=True,
        help='path of configs directory')
    parser.add_argument('-m', '--model-type', choices=['AE', 'VQ-VAE'], required=True,
        help='type of model [AE or VQ-VAE]')
    parser.add_argument('-n', '--n-image', type=int, default=25,
        help='number of output images')
    parser.add_argument('--output-folder', type=str, default='rf_imgs',
        help='name of the output folder')
    parser.add_argument('--best', type=str, default='',
        help='')
    args = parser.parse_args()

    main(args)