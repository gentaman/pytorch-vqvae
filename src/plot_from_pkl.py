import pickle
import os
from glob import glob
import re

import matplotlib.pyplot as plt


def get_plotargs(name):
    label_name = ''
    if 'k32' in name:
        label_name = label_name + 'k32'
        c1 = 255
        c2 = 80
        vq = True
    elif 'k4' in name:
        label_name = label_name + 'k4'
        c1 = 80
        c2 = 255
        vq = True
    else:
        c1 = 255
        c2 = 0
        vq = False
    
    if 'no_recon' in name:
        if vq:
            label_name = ' '.join(['VQ-CNN', label_name])
        else:
            label_name = ' '.join(['CNN', label_name])
        ls = '--'
        r = c1
        g = c2
        b = 0
#         r = 0
#         g = c1
#         b = c2
        color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
    elif 'no_pred' in name:
        if vq:
            label_name = ' '.join(['VQ-VAE', label_name])
        else:
            label_name = ' '.join(['AE', label_name])
        ls = ':'
        r = c1
        g = c2
        b = 0
#         r = c2
#         g = 0
#         b = c1
        color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
    else:
        if vq:
            label_name = ' '.join(['multi-VQ-VAE', label_name])
        else:
            label_name = ' '.join(['multi-AE', label_name])
        ls = '-'
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

def main(scalar_path, out_dir):

    with open(scalar_path, 'rb') as f:
        scalar_data = pickle.load(f)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fontsize = 24
    plt.rcParams["font.size"] = fontsize

    num_epoch = 100
    figsize = (16, 9)

    figs_num = {}
    file_keys = sorted(list(scalar_data.keys()))
    y_min_max = {}

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
                plotargs[label] = file_key
            ax.plot(data, alpha=0.6, **plotargs)
            
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
        ax.legend(fontsize=int(fontsize * 2 / 3))
        
        out_path = os.path.join(out_dir, '{}.png'.format(key.replace('/', '-')))
        plt.savefig(out_path)
        plt.close()
    

if __name__ == '__main__':
    import sys

    main(sys.argv[1], sys.argv[2])
