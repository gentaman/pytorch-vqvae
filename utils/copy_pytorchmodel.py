'''
    To copy model parameters to a similar model.
'''


import sys
from collections import OrderedDict
import copy

import torch
from torch import nn


def copy_model(src_model, dst_model, layer_names=None, verbose=-1):
    '''
        PARAMETER:
        src_model: source model
        dst_model: model to be copied
        layer_names: define the name of the layer to be copied.
                     if None then all layers to be copied.
        verbose: if verbose > -1 then print informations.

        RETURN:
        dst_model: model to be copied

        First, try dst_model.load_state_dict(src_state_dict).
        If catch a RuntimeError, make an effort copy parameters of src_model to dst_model.
    '''
    # src_model == PATH
    if isinstance(src_model, str):
        src_state_dict = torch.load(src_model)
    # src_model == Module
    elif isinstance(src_model, nn.Module):
        src_state_dict = src_model.state_dict()
    elif isinstance(src_model, OrderedDict):
        src_state_dict = src_model
    else:
        raise TypeError('Undefine {} type. source model'.format(type(src_model)))
    
    assert isinstance(dst_model, nn.Module)

    dst_state_dict = dst_model.state_dict()
    count = 0
    if layer_names is not None:
        for key in dst_state_dict:
            if 'module.' in key:
                layer_name = key[len('module.'):key[len('module.'):].find('.')+len('module.')]
            else:
                layer_name = key[:key.find('.')]
                
            if layer_name not in layer_names:
                if verbose > -1:
                    print('skip: {}'.format(key))
                src_state_dict[key] = dst_state_dict[key]
            else:
                if verbose > -1:
                    print('replace: {} {}'.format(key, count))
                    count += 1

    try:
        dst_model.load_state_dict(src_state_dict)

    except RuntimeError as e:
        string_e = str(e)
        # process for missing key 
        key_error_msgs = [
            'Missing key(s) in state_dict:',
            'Unexpected key(s) in state_dict: ',
        ]
        mismatch_key = 'size mismatch'

        def find_key_error(key):
            # find start index
            s_index = string_e.find(key)
            size_mismatch_index = -1
            if s_index == -1:
                size_mismatch_index = string_e.find(mismatch_key)
                return list(),  size_mismatch_index
            s_index += len(key)
            # find end index
            for key in key_error_msgs + [mismatch_key]:
                e_index = string_e[s_index:].find(key)
                if e_index > -1:
                    e_index += s_index
                    if mismatch_key == key:
                        size_mismatch_index = e_index
                    break
            # print(string_e[s_index:e_index])

            # remove end '.'
            if e_index == -1:
                e_index = -2
            else:
                tmp_index = string_e[s_index:e_index].rfind('.')
                if tmp_index == -1:
                    raise ValueError('Not found "." from {}'.format(string_e[:e_index]))
                e_index = tmp_index + s_index
            string = string_e[s_index:e_index]
            keys = string.replace('"', '').replace(' ', '').split(',')
            return keys, size_mismatch_index
        
        missing_keys, e_index0  = find_key_error(key_error_msgs[0])
        unexpected_keys, e_index1 = find_key_error(key_error_msgs[1])

        def get_mismatch_keys(string_e):
            mismatch_keys = {}
            mismatches = string_e.split('size mismatch')
            # remove a empty element
            mismatches = mismatches[1:]
            for m in mismatches:
                s_index = m.find('for ') + len('for ')
                e_index = m.find(':')
                mismatch_key = m[s_index:e_index]
                s_index = m.rfind('torch.Size(') + len('torch.Size(')
                e_index = m.rfind(').')
                shape = m[s_index:e_index]
                
                if shape[0] == '[':
                    shape = shape[1:-1]
                    # print(shape)
                    if ',' in shape:
                        shape = shape.split(',')
                        shape = list(map(lambda x: int(x), shape))
                    else:
                        shape = list((int(shape), ))
                else:
                    # TODO
                    pass
                mismatch_keys[mismatch_key] = shape
            return mismatch_keys

        # process for size mismatch key
        if e_index0 > -1:
            mismatch_keys = get_mismatch_keys(string_e[e_index0:])
        elif e_index1 > -1:
            mismatch_keys = get_mismatch_keys(string_e[e_index1:])
        else:
            mismatch_keys = {}
        
        for key in missing_keys:
            if verbose > -1:
                print('Using dst model, Missing Key; {}'.format(key))
            # using dst_model value
            src_state_dict[key] = dst_state_dict[key]
        
        for key in unexpected_keys:
            if verbose > -1:
                print('remove Unexpected Key; {}'.format(key))
            # remove key; using random value
            del src_state_dict[key]
            
        for key in mismatch_keys:
            if verbose > -1:
                print('Using dst model, Size Mismatch Key; {}'.format(key))
            # using dst_model value
            src_state_dict[key] = dst_state_dict[key]


        dst_model.load_state_dict(src_state_dict)

    return dst_model
