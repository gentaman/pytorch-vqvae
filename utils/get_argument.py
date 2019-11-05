import argparse
import multiprocessing as mp
import sys

def parse_config(config_path):
    with open(config_path, 'r') as f:
        data = f.read()
        data = data.split('\n')
        # remove python script name
        # result = ' '.join(data[1:])
        result = data[1:]
        
    return result

def update(actions, config, argv):
    for act in actions:
        opt_string = list(set(act.option_strings) & set(argv))
        c_find = list(set(act.option_strings) & set(config))
        if len(opt_string) > 0:
            a_index = argv.index(opt_string[0])
            if isinstance(act, argparse._StoreAction):
                if len(c_find) > 0:
                    c_index = config.index(c_find[0])
                    config[c_index] = argv[a_index]
                    config[c_index+1] = argv[a_index+1]
                else:
                    config.append(argv[a_index])
                    config.append(argv[a_index+1])

            elif isinstance(act, argparse._StoreFalseAction) or isinstance(act, argparse._StoreTrueAction):
                if len(c_find) > 0:
                    pass
                else:
                    config.append(argv[a_index])
    return config
                



def get_args(description='VQ-VAE', return_parser=False, inputs=None):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--config', type=str, default=None,
        help='config file path')

    if 'Performance' in description:
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
        parser.add_argument('--output-folder', type=str, default='preform',
            help='name of the output folder (default: preform)')
        parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
            help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
        parser.add_argument('--device', type=str, default='cpu',
            help='set the device (cpu or cuda, default: cpu)')

        # Visualization
        parser.add_argument('--off-eachnorm', dest='each_normalize', action='store_false',
            help='disable noramalization of visualizing image')
        parser.add_argument('--plot-center', dest='project_center', action='store_true',
            help='visualizing codebook with one-hot center future map')

    elif 'VQ-VAE with prediction' in description:
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
        parser.add_argument('--off-bias', dest='bias', action='store_false',
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

    elif 'AE with prediction' in description:
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
            help='filename containing the preditor')
        parser.add_argument('--gap', action='store_true',
            help='add GAP')
        parser.add_argument('--resblock-transpose', action='store_true',
            help='apply conv transpose to ResBlock')


        # Latent space
        parser.add_argument('--hidden-size', type=int, default=256,
            help='size of the latent vectors (default: 256)')
        parser.add_argument('--k', type=int, default=512,
            help='number of latent vectors (default: 512)')
        parser.add_argument('--prior', type=str, default='Uniform',
            help='name of prior distribution (default: Uniform)')
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


        # Miscellaneous
        parser.add_argument('--root', type=str, default='.',
            help='name of the root of the output folder (default: .)')
        parser.add_argument('--output-folder', type=str, default='vqvae',
            help='name of the output folder (default: vqvae)')
        parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
            help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
        parser.add_argument('--device', type=str, default='cpu',
            help='set the device (cpu or cuda, default: cpu)')

    if return_parser:
        return parser
    
    if inputs is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(inputs)
    
    if args.config is None:
        return args
    else:
        config = parse_config(args.config)
        config = update(parser._actions, config, sys.argv[1:])
        args, unkown = parser.parse_known_args(config)
        if len(unkown) > 0:
            print('unkown: ', unkown)
        return args
