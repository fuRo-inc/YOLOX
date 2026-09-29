#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import itertools
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler


class YoloBatchSampler(torchBatchSampler):
    """
    This batch sampler will generate mini-batches of (mosaic, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will turn on/off the mosaic aug.
    """

    def __init__(self, *args, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mosaic = mosaic

    def __iter__(self):
        for batch in super().__iter__():
            yield [(self.mosaic, idx) for idx in batch]


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size


class WeightedInfiniteSampler(Sampler):
    """An infinite, replacement sampler with per-index sampling weights.

    Every DDP rank consumes a disjoint stride of one deterministic global
    stream.  As with :class:`InfiniteSampler`, this keeps ranks in sync while
    retaining the usual replacement semantics (the same image can occur more
    than once in an epoch).
    """

    def __init__(self, weights, seed: Optional[int] = 0, rank=0, world_size=1):
        self._weights = torch.as_tensor(weights, dtype=torch.double, device="cpu")
        if self._weights.ndim != 1 or self._weights.numel() == 0:
            raise ValueError("weights must be a non-empty one-dimensional sequence")
        if not torch.isfinite(self._weights).all() or (self._weights < 0).any():
            raise ValueError("weights must be finite and greater than or equal to zero")
        if self._weights.sum().item() <= 0:
            raise ValueError("the sum of weights must be greater than zero")

        self._size = self._weights.numel()
        self._seed = int(seed or 0)
        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size
        if not 0 <= self._rank < self._world_size:
            raise ValueError("rank must be in [0, world_size)")

    def __iter__(self):
        yield from itertools.islice(
            self._infinite_indices(), self._rank, None, self._world_size
        )

    def _infinite_indices(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed)
        # Drawing modest chunks avoids per-index torch calls without retaining
        # an epoch-sized index list.  This is deliberately CPU-only.
        while True:
            yield from torch.multinomial(
                self._weights,
                num_samples=1024,
                replacement=True,
                generator=generator,
            ).tolist()

    def __len__(self):
        return self._size // self._world_size
