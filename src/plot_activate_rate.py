import os
import pickle

from tqdm import tqdm
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

from codebook_effect import damage_latent_code
import sys
sys.path.append('..')
from modules.modules import VectorQuantizedVAE, AE
from modules.functions import Classifier
from datasets.datasets import get_dataset
from utils.get_argument import get_args, parse_config
del sys.path[-1]


def plot_activate_rate(count_ks, dnames):
    plt.figure(figsize=(16, 16))
    plt.boxplot()

def main(main_args):
    dnames = [
    'mnist_im32_gap_k16_e100',
    'mnist_im32_gap_k32_e100',
    'mnist_im32_gap_k64_e100',
    'mnist_im32_gap_k128_e100',
    'mnist_im32_gap_k256_e100',
    'mnist_im32_gap_k512_e100'
    ]
    n = len(dnames)
    count_ks = [list() for i in range(n)]
    
    for configs in main_args.configs:
        configs_path = configs
        # dnames  = sorted(os.listdir(configs_path))
        for i, dname in enumerate(dnames):
            print(dname)
            config_path = os.path.join(configs_path, dname, 'config')

            args = get_args(description='VQ-VAE', inputs=['--config', config_path])
            device = args.device
            image_size = args.image_size
            dataset = args.dataset
            data_folder = args.data_folder
            result = get_dataset(dataset, data_folder, image_size=image_size)
            test_dataset = result['test']
            num_channels = result['num_channels']
            hidden_size = args.hidden_size

            # TODO: no gap, BN, bias - model
            if hasattr(args, 'k'):
                k = args.k
                model = VectorQuantizedVAE(num_channels, hidden_size, k, pred=True, transpose=args.resblock_transpose).to(device)
                n_in = hidden_size
                n_out = 10
                predictor = Classifier(n_in, n_out, gap=args.gap).to(device)
            else:
                model = AE(num_channels, hidden_size, pred=True, transpose=args.resblock_transpose).to(device)
                n_in = hidden_size
                n_out = 10
                predictor = Classifier(n_in, n_out, gap=args.gap).to(device)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=main_args.n_image, shuffle=False)
            images, labels = next(iter(test_loader))

            for n_cv in range(args.kfold):
                m_name = 'models_{}'.format(n_cv)
                model_path = os.path.join(args.root, m_name, args.output_folder, 'best.pt')
                print('load model ==> {}'.format(model_path))
                with open(model_path, 'rb') as f:
                    model.load_state_dict(torch.load(f))

                model_path = os.path.join(args.root, m_name, args.output_folder, 'best_predictor.pt')
                print('load model ==> {}'.format(model_path))
                with open(model_path, 'rb') as f:
                    predictor.load_state_dict(torch.load(f))
                
                count_k = [0 for _ in range(args.k)]     
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    rec_imgs, _, latents, _ = model(images)
                    preds = predictor(latents)
                    
                    for k in range(args.k):
                        count_k[k] += torch.sum(latents == k).item()
                
                correct_index = preds.argmax(1) == labels
                result = damage_latent_code(model, predictor, images[correct_index], labels[correct_index], device=args.device)

            count_ks[i].append(count_k)
    out_dir = os.path.join(os.path.dirname(configs_path), main_args.output_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('write == > {}'.format(out_dir))
    plot_activate_rate(count_ks, dnames)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot codebook effect')
    parser.add_argument('-c', '--configs', type=str, nargs='+', required=True,
        help='path of configs directory')
    parser.add_argument('-m', '--model-type', choices=['AE', 'VQ-VAE'], required=True,
        help='type of model [AE or VQ-VAE]')
    parser.add_argument('-n', '--n-image', type=int, default=25,
        help='number of output images')
    parser.add_argument('--output-folder', type=str, default='activate_rate',
        help='name of the output folder')
    args = parser.parse_args()
    
    main(args)