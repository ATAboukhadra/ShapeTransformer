##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, December 2022
##############################################################################
from argparse import ArgumentParser

from torchdata.datapipes.iter import IterDataPipe

from datapipes.sequence_pipeline import SequencePipelineCreator
from datapipes.utils.dispatcher import Dispatcher
from datapipes.utils.remove_wrapper import remove_wrapper
from datapipes.decoders.image_decoder import ImageDecoder
import torch
import numpy as np
from torch.utils.data.datapipes.datapipe import DataChunk
from dataset.arctic_dataset import ArcticDataset
import time 
from torchvision.transforms.functional import resize
from datapipes.utils.dataset_path_utils import get_component_name

class ArcticDecoder():
    def __init__(self, root, objects_root, device):
        self.dataset = ArcticDataset(root, objects_root, device, iterable=True)
    
    def __call__(self, subset_id, seq_id, sample_id):
        splits = seq_id.split('_')
        subject = splits[0]
        seq_id = '_'.join(splits[1:-1])
        camera = splits[-1]
        key = '/'.join([subject, seq_id, camera, sample_id])
        data_dict = self.dataset.get_anno(key)

        new_components = []
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                component_name = k+'.npy'
            elif isinstance(v, torch.Tensor):
                component_name = k+'.pt'
            elif isinstance(v, str):
                component_name = k+'.txt'
            else:
                component_name = k+'.bool'
            new_components.append((component_name, v))

        return new_components

def temporal_batching(batch):
    data_dict = {}
    bs = len(batch)
    ws = len(batch[0])

    for i, window in enumerate(batch):
        for j, sample in enumerate(window):
            for path, data in sample:
                comp = get_component_name(path)
                if comp not in data_dict.keys():
                    if isinstance(data, torch.Tensor) and comp != 'rgb':
                        data_dict[comp] = torch.zeros((bs, ws, *data.shape), dtype=data.dtype)
                    elif isinstance(data, np.ndarray):
                        data_dict[comp] = np.zeros((bs, ws, *data.shape), dtype=data.dtype)
                    else:
                        data_dict[comp] = [[[None]] * ws] * bs

                data_dict[comp][i][j] = data
    
    data_dict['rgb'] = [torch.stack(image_batch, dim=0) for image_batch in data_dict['rgb']]

    return data_dict

def resize_sample(path, sample):
    resized_img = resize(sample, 500, antialias=True)
    return resized_img

def filter_fn(sample):
    valids = [data for s in sample for path, data in s if 'valid' in path]
    return all(valids)

def decode_dataset(pipe: IterDataPipe, root, objects_root, device, arctic_decoder=None):
    """ Decodes the components of the dataset "dataset". """
    # TODO map components to decoding functions
    decoder_map = {"rgb": ImageDecoder('torch', decode_truncated_images=True)}
    # apply the decoding functions to each sample
    pipe = pipe.map(fn=Dispatcher(decoder_map))
    pipe = pipe.map(fn=Dispatcher({"rgb": resize_sample}, 'decoded', 'resized'))
    pipe = pipe.map(fn=remove_wrapper)
    pipe = pipe.filter(filter_fn=filter_fn)
    # pipe = pipe.map(fn=resize_sample)
    # arctic_decoder = ArcticDecoder(root, objects_root, device) if arctic_decoder is None else arctic_decoder
    # pipe = pipe.map(fn=arctic_decoder)
    return pipe

def create_pipe(in_dir, objects_root, subset, device, sliding_window_size, factory=None, arctic_decoder=None):

    # Make sure to create the factory only once. It reads the metadata file at construction time.
    if factory is None: factory = SequencePipelineCreator(in_dir)

    # Make the shuffle buffer a multiple of the shard size. The multiplier may be chosen according to the batch size.
    # multiplier = 1
    # shard_size = factory.get_average_shard_sample_count(subset)
    sample_count = factory.get_sample_count(subset)
    # Make an educated guess on a good size for the shuffle buffer using the meta-data.
    # shuffle_buffer_size = int(multiplier * shard_size)
    
    shuffle_buffer_size = 2 # This is now the number of sequences in the buffer

    # Using the metadata created in the conversion process, the streaming pipeline can be created automatically.
    arctic_decoder = ArcticDecoder(in_dir, objects_root, device) if arctic_decoder is None else arctic_decoder
    pipe = factory.create_datapipe(subset, shuffle_buffer_size, shuffle_shards=subset == "train", temporal_sliding_window_size=sliding_window_size, add_component_fn=arctic_decoder)
    # Decode the components of the dataset. Placing it in a function makes it reusable.
    pipe = decode_dataset(pipe, in_dir, objects_root, device, arctic_decoder)
    # pipe = factory.add_temporal_windowing_and_shuffling(pipe)
    # pipe = pipe.map(fn=arctic_decoder)

    # pipe.map(fn=batch_samples)
    return pipe, sample_count, arctic_decoder, factory
