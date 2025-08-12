from partcrafter_src.utils.typing_utils import *

import torch

from .objaverse_part import ObjaversePartDataset, BatchedObjaversePartDataset

# Copied from https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/loader.py
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler
        if isinstance(self.sampler, torch.utils.data.sampler.BatchSampler):
            self.batch_size = self.sampler.batch_size
            self.drop_last = self.sampler.drop_last

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def yield_forever(iterator: Iterator[Any]):
    while True:
        for x in iterator:
            yield x