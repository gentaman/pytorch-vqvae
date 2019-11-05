import pickle
import os
from glob import glob
import re

import matplotlib.pyplot as plt


def get_plotargs(name, train_vs_test=False):
    label_name = ''
    if 'k256' in name:
        label_name = label_name + 'k128'
        c1 = 224
        c2 = 224
        c3 = 80
        vq = True
    elif 'k128' in name:
        label_name = label_name + 'k128'
        c1 = 224
        c2 = 80
        c3 = 80
        vq = True
    elif 'k32' in name:
        label_name = label_name + 'k32'
        c1 = 80
        c2 = 224
        c3 = 80
        vq = True
    elif 'k4' in name:
        label_name = label_name + 'k4'
        if 'k4-2' in name:
            c1 = 224
            c2 = 80
            c3 = 224
        else:
            c1 = 80
            c2 = 80
            c3 = 224
        vq = True

    else:
        c1 = 0
        c2 = 0
        c3 = 0
        vq = False
            
    if 'no_recon' in name:
        if vq:
            label_name = ' '.join(['VQ-CNN', label_name])
        else:
            label_name = ' '.join(['CNN', label_name])
        ls = '--'
    elif 'no_pred' in name:
        if vq:
            label_name = ' '.join(['VQ-VAE', label_name])
        else:
            label_name = ' '.join(['AE', label_name])
        ls = ':'
    else:
        if vq:
            label_name = ' '.join(['multi-VQ-VAE', label_name])
        else:
            label_name = ' '.join(['multi-AE', label_name])
        ls = '-'

    if 'fashion-mnist' in name:
        r = c3
        g = c1
        b = c2
        color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        label_name = ' '.join([label_name, 'fashion-MNIST'])
    elif 'mnist' in name:
        r = c1
        g = c2
        b = c3
        color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        label_name = ' '.join([label_name, 'MNIST'])
    else:
        r = c1
        g = c2
        b = 0
        color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
    
    
    args = {
        'linestyle': ls,
        'color': color,
        'label': label_name
    }
    return args

def global_summary(scalar_data, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fontsize = 24
    plt.rcParams["font.size"] = fontsize
    num_epoch = 100
    figsize = (16, 9)

    figs_num = {}
    y_min_max = {}

    def rm_key(key):
        if 'k4-2' in key:
            return True
        
        if 'k4' in key:
            return False
        
        return True

    file_keys = sorted(list(filter(rm_key, scalar_data.keys())))
    for file_key in file_keys:
        scalar = scalar_data[file_key]
        plotargs = get_plotargs(file_key)

        for data_key in scalar:
            key_without_train_test = re.sub('train|test|val', '', data_key)
            if data_key not in figs_num:
                fig = plt.figure(figsize=figsize)
                figs_num[data_key] = fig.number
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title(data_key)
            else:
                fig = plt.figure(figs_num[data_key])
                ax = fig.axes[0]
            data = scalar[data_key]
            if 'train' in data_key:
                 data = data.reshape(num_epoch, -1).mean(1)
            if 'label' not in plotargs:
                plotargs["label"] = file_key
            ax.plot(data, alpha=0.7, **plotargs)
            
            if key_without_train_test not in y_min_max:
                y_min_max[key_without_train_test] = [data.min(), data.max()]
            else:
                if y_min_max[key_without_train_test][0] > data.min():
                    y_min_max[key_without_train_test][0] = data.min()
                if y_min_max[key_without_train_test][1] < data.max():
                    y_min_max[key_without_train_test][1] = data.max()
                

    for key in figs_num:
        print(key)
        key_without_train_test = re.sub('train|test|val', '', key)
        fig = plt.figure(figs_num[key])
        ax = fig.axes[0]
        e0 = 1
        e1 = 1
        if 'loss' in key:
            ax.set_yscale('log')
            e0 = 1e-10
            e1 = y_min_max[key_without_train_test][1]
        ax.set_ylim(y_min_max[key_without_train_test][0]-e0, y_min_max[key_without_train_test][1]+e1)
        ax.grid()
        ax.legend(fontsize=int(fontsize * 3 / 5), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        
        
        out_path = os.path.join(out_dir, '{}.png'.format(key.replace('/', '-')))
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

def train_test_summary(scalar_data, out_dir, adjust_y_scale=True, **kwargs):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fontsize = 24
    plt.rcParams["font.size"] = fontsize
    num_epoch = 100
    figsize = (16, 9)

    figs_num = {}
    y_min_max = {}
    file_keys = sorted(list(scalar_data.keys()))
    for file_key in file_keys:
        scalar = scalar_data[file_key]
        plotargs = get_plotargs(file_key)
        if 'label' not in plotargs:
            plotargs["label"] = file_key
        label_name = plotargs["label"]

        for data_key in scalar:
            key_without_train_test = re.sub('/train|/test|/val', '', data_key)
            fig_key = '/'.join([label_name, key_without_train_test])
            if fig_key not in figs_num:
                fig = plt.figure(figsize=figsize)
                figs_num[fig_key] = fig.number
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title(fig_key)
            else:
                fig = plt.figure(figs_num[fig_key])
                ax = fig.axes[0]
            data = scalar[data_key]
            if 'train' in data_key:
                data = data.reshape(num_epoch, -1).mean(1)
                plotargs['linestyle'] = ':'
                plotargs['label'] = 'train'
            if 'test' in data_key:
                plotargs['linestyle'] = '-'
                plotargs['label'] = 'test'
            
            ax.plot(data, alpha=0.7, **plotargs)
            
            if key_without_train_test not in y_min_max:
                y_min_max[key_without_train_test] = [data.min(), data.max()]
            else:
                if y_min_max[key_without_train_test][0] > data.min():
                    y_min_max[key_without_train_test][0] = data.min()
                if y_min_max[key_without_train_test][1] < data.max():
                    y_min_max[key_without_train_test][1] = data.max()
                

    y_keys = list(y_min_max.keys())
    for key in figs_num:
        print(key)
        key_without_train_test = list(filter(lambda x: x in key, y_keys))[0]
        fig = plt.figure(figs_num[key])
        ax = fig.axes[0]
        e0 = 1
        e1 = 1
        if 'loss' in key:
            ax.set_yscale('log')
            e0 = 1e-10
            e1 = y_min_max[key_without_train_test][1]
        if adjust_y_scale:
            ax.set_ylim(y_min_max[key_without_train_test][0]-e0, y_min_max[key_without_train_test][1]+e1)
        ax.grid()
        ax.legend(fontsize=int(fontsize * 3 / 5), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        
        
        out_path = os.path.join(out_dir, '{}.png'.format(key.replace('/', '-').replace(' ', '-')))
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()



def main(scalar_path, out_dir, **kwargs):

    if isinstance(scalar_path, list):
        scalar_data = {}
        for s_path in scalar_path:
            with open(s_path, 'rb') as f:
                scalar_data.update(pickle.load(f))

    elif isinstance(scalar_path, str):
        with open(scalar_path, 'rb') as f:
            scalar_data = pickle.load(f)

    # global summary
    global_summary(scalar_data, os.path.join(out_dir, 'global'))

    # train-test summary
    train_test_summary(scalar_data, os.path.join(out_dir, 'train-vs-test'), **kwargs)

    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot some graphes')

    parser.add_argument('-s', '--scalars', type=str, nargs='+',
        help='name of the scalar data')
    parser.add_argument('--output-folder', type=str, default=None,
        help='name of the output folder')
    parser.add_argument('--adjust-y-scale', action='store_true',
        help='adjust y scale over same kind data')
    args = parser.parse_args()

    if args.output_folder is None:
        dirname = os.path.dirname(args.scalars[0])
        dir_path = os.path.dirname(dirname)

        args.output_folder = os.path.join(dir_path, 'imgs')
        
    main(args.scalars, args.output_folder, 
        adjust_y_scale=args.adjust_y_scale)
