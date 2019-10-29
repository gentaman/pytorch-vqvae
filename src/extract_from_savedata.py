
import os
from glob import glob
import pickle

import numpy as np
import cv2
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# root_path = '/home/genta/data2/20191023_transpose_vqvae/logs/'
# out_dir = './out'

def main(root_path, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    datas = {}
    for fname in os.listdir(root_path):
        path = os.path.join(root_path, fname)
        event_path = glob(os.path.join(path, 'events*'))[0]
        print('load ==> {}'.format(event_path))
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()

        # save scalar data
        scalars = {}
        for s_tag in event_acc.Tags()['scalars']:
            events = event_acc.Scalars(s_tag)
            scalars[s_tag] = np.asarray([event.value for event in events])
        
        datas[fname] = scalars
        
        out_dir_fname = os.path.join(out_dir, fname)
        if not os.path.exists(out_dir_fname):
            os.makedirs(out_dir_fname)
        # save image data
        for i_tag in event_acc.Tags()['images']:
            events = event_acc.Images(i_tag)
            tag_name = i_tag.replace('/', '_')
            for index, event in enumerate(events):
                s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
                image = cv2.imdecode(s, cv2.IMREAD_COLOR)
                out_name = '{}_{:03}.jpg'.format(tag_name, event.step)
                outpath = os.path.join(out_dir_fname, out_name)
                print('write image ==> {}'.format(outpath))
                cv2.imwrite(outpath, image)

    pkl_path = os.path.join(out_dir, 'scalars_data.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(datas, f)

if __name__ == '__main__':
    import sys

    root_path = sys.argv[1]
    out_dir = sys.argv[2]

    main(root_path, out_dir)

