##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, December 2022
##############################################################################
from argparse import ArgumentParser

from torchdata.datapipes.iter import IterDataPipe

from datapipes.sequence_pipeline import SequencePipelineCreator
from datapipes.utils.decoder import Decoder
from datapipes.utils.image_decoder import ImageDecoder
import torch
import numpy as np
from torch.utils.data.datapipes.datapipe import DataChunk
from dataset.arctic_dataset import ArcticDataset

class ArcticDecoder():
    def __init__(self, root, objects_root, device):
        self.dataset = ArcticDataset(root, objects_root, device, iterable=True)
    
    def decode_sample(self, sample: DataChunk):
        splits = sample[0][0].split('/')

        key = splits[-1].split('.')[:2]
        seq_id = key[0].split('_')

        key = '/'.join([seq_id[0], '_'.join(seq_id[1:-1]), seq_id[-1], key[1]]) 
        
        data_dict = self.dataset.get_anno(sample[0][1], key)
        return data_dict


    def __call__(self, sample: DataChunk):
        seq_samples = [self.decode_sample(s) for s in sample]

        non_changing_keys = ['left_shape', 'right_shape', 'object_name', 'label', 'cam_int']
        temporal_sample = batch_samples(seq_samples, non_changing_keys=non_changing_keys, rigged_rgb=False)
        
        return temporal_sample

def batch_samples(samples, non_changing_keys=None, rigged_rgb=True):
    # Concatenate the tensors for each key across all dictionaries

    keys = list(samples[0].keys())
    samples_dict = {}
    for key in keys:
        if non_changing_keys is not None and key in non_changing_keys:
            samples_dict[key] = samples[0][key]
        elif rigged_rgb and key == 'img':
            samples_dict[key] = [s[key] for s in samples]
        elif isinstance(samples[0][key], torch.Tensor):
            samples_dict[key] = torch.cat([s[key].unsqueeze(0) for s in samples], dim=0)
        elif isinstance(samples[0][key], np.ndarray):
            samples_dict[key] = torch.cat([torch.tensor(s[key]).unsqueeze(0) for s in samples], dim=0)
        else:
            samples_dict[key] = [d[key] for d in samples]

    return samples_dict


def decode_dataset(pipe: IterDataPipe, root, objects_root, device, arctic_decoder=None):
    """ Decodes the components of the dataset "dataset". """
    # TODO map components to decoding functions
    decoder_map = {"rgb": ImageDecoder('rgb8')}
    # apply the decoding functions to each sample
    pipe = pipe.map(fn=Decoder(decoder_map))
    arctic_decoder = ArcticDecoder(root, objects_root, device) if arctic_decoder is None else arctic_decoder
    pipe = pipe.map(fn=arctic_decoder)
    return pipe, arctic_decoder


def create_pipe(in_dir, objects_root, subset, device, sliding_window_size, factory=None, arctic_decoder=None):

    # Make sure to create the factory only once. It reads the metadata file at construction time.
    if factory is None: factory = SequencePipelineCreator(in_dir)

    # Make the shuffle buffer a multiple of the shard size. The multiplier may be chosen according to the batch size.
    multiplier = 2
    shard_size = factory.get_average_shard_sample_count(subset)
    # Make an educated guess on a good size for the shuffle buffer using the meta-data.
    shuffle_buffer_size = int(multiplier * shard_size)
    # Using the metadata created in the conversion process, the streaming pipeline can be created automatically.
    pipe = factory.create_datapipe(subset, shuffle_buffer_size, shuffle_shards=subset == "train", temporal_sliding_window_size=sliding_window_size)
    # Decode the components of the dataset. Placing it in a function makes it reusable.
    pipe, decoder = decode_dataset(pipe, in_dir, objects_root, device, arctic_decoder)

    return pipe, decoder, factory
