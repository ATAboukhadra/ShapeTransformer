##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, December 2022
##############################################################################
from argparse import ArgumentParser
import time

from torchdata.datapipes.iter import IterDataPipe

from datapipes.base_pipeline import BasePipelineCreator
from datapipes.sequence_pipeline import SequencePipelineCreator
from datapipes.utils.decoder import Decoder
from dataset.DexYCB_utils import decode_depth, decode_labels, decode_rgb, preprocess, dexycb_dataset_dict, preprocess_sequential
from dataset.DexYCB_dataset import DexYCBDataset


def decode_dataset(pipe: IterDataPipe, sequential=False):
    """ Decodes the components of the dataset "dataset". """
    # TODO map components to decoding functions
    decoder_map = {"depth": decode_depth,
                   "color": decode_rgb,
                   "labels": decode_labels}
    # apply the decoding functions to each sample

    pipe = pipe.map(fn=Decoder(decoder_map))
    if sequential:
        pipe = pipe.map(fn=preprocess_sequential)
    else:
        pipe = pipe.map(fn=preprocess)

    return pipe

def create_pipe(in_dir, subset, args, sequential=False, factory=None):
    # DexYCB specific step
    global dexycb_dataset_dict
    dexycb_dataset_dict[subset] = DexYCBDataset(args.meta_root, subset)

    # Make sure to create the factory only once. It reads the metadata file at construction time.
    if factory is None:
        if sequential:
            factory = SequencePipelineCreator(in_dir)
        else:
            factory = BasePipelineCreator(in_dir)

    # Make the shuffle buffer a multiple of the shard size. The multiplier may be chosen according to the batch size.
    multiplier = 2
    shard_size = factory.get_average_shard_sample_count(subset)
    # Make an educated guess on a good size for the shuffle buffer using the meta-data.
    shuffle_buffer_size = int(multiplier * shard_size)
    # Using the metadata created in the conversion process, the streaming pipeline can be created automatically.
    if sequential:
        pipe = factory.create_datapipe(subset, shuffle_buffer_size, shuffle_shards=True, temporal_sliding_window_size=args.window_size)    
    else:
        pipe = factory.create_datapipe(subset, shuffle_buffer_size, shuffle_shards=subset == "train")
    # Decode the components of the dataset. Placing it in a function makes it reusable.
    pipe = decode_dataset(pipe, sequential=sequential)

    return dexycb_dataset_dict[subset], pipe, factory
