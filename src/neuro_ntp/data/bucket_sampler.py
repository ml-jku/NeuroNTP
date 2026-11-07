# bucket_sampler.py
import math, random
from collections import defaultdict
from typing import Callable, Dict, Iterator, List, Tuple
import torch
from torch.utils.data import Sampler

class KeyBucketSampler(Sampler[int]):
    """
    Groups dataset indices by a key (key_fn(idx)) and yields a *flat sequence of indices*
    where buckets are interleaved. DDP-aware via (num_replicas, rank).
    Wrap with torch.utils.data.BatchSampler to form mini-batches.
    """

    def __init__(
        self,
        dataset,
        key_fn: Callable[[int], Tuple],
        *,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        temperature: float = 1.0,  # 1.0 ~ proportional to bucket size; ->0 ~ round-robin
        rebuild_buckets_each_epoch: bool = False,  # set True if the view can change
    ):
        super().__init__()
        self.dataset = dataset
        self.key_fn = key_fn
        self.shuffle = bool(shuffle)
        self.base_seed = int(seed)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.temperature = max(1e-8, float(temperature))
        self.rebuild_buckets_each_epoch = bool(rebuild_buckets_each_epoch)

        self._epoch = 0
        self._buckets: Dict[Tuple, List[int]] = {}
        self._bucket_keys: List[Tuple] = []
        self._shard_indices: List[int] = []

        # Build once initially; can be rebuilt each epoch if needed
        self._build_buckets()

        # Prepare shard for epoch 0
        self._prepare_shard_for_epoch(0)

    # ---------- public API ----------

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if self.rebuild_buckets_each_epoch:
            self._build_buckets()
        self._prepare_shard_for_epoch(self._epoch)

    def __len__(self) -> int:
        # Always reflect the actual per-rank plan
        return len(self._shard_indices)

    def __iter__(self) -> Iterator[int]:
        # If someone forgot to call set_epoch on epoch>0, make sure we’re consistent
        if self._epoch == 0 and not self._shard_indices:
            self._prepare_shard_for_epoch(0)
        return iter(self._shard_indices)

    # ---------- internals ----------

    def _build_buckets(self) -> None:
        buckets: Dict[Tuple, List[int]] = defaultdict(list)
        for idx in range(len(self.dataset)):
            buckets[self.key_fn(idx)].append(idx)
        # drop empties just in case
        self._buckets = {k: v for k, v in buckets.items() if v}
        self._bucket_keys = list(self._buckets.keys())

    def _prepare_shard_for_epoch(self, epoch: int) -> None:
        rng = random.Random(self.base_seed + int(epoch))

        # Copy & (optionally) shuffle inside each bucket
        bucket_lists: Dict[Tuple, List[int]] = {}
        for k, idxs in self._buckets.items():
            li = list(idxs)
            if self.shuffle:
                rng.shuffle(li)
            bucket_lists[k] = li

        if not bucket_lists:
            self._shard_indices = []
            return

        keys = list(bucket_lists.keys())

        # Build an interleaving schedule across buckets
        if self.shuffle:
            sizes = [len(bucket_lists[k]) for k in keys]
            # weight per bucket ~ size^temperature
            weights = [max(1e-8, s ** self.temperature) for s in sizes]
            total = sum(weights)
            # repeats proportional to weight, but ensure at least 1 visit
            total_repeats = max(len(keys), int(round(total)))
            repeats = [max(1, int(round(w / total * total_repeats))) for w in weights]
            sched = [k for k, r in zip(keys, repeats) for _ in range(r)]
            rng.shuffle(sched)
        else:
            sched = keys  # deterministic

        # Interleave indices following sched
        cursors = {k: 0 for k in bucket_lists}
        flat: List[int] = []
        progressed = True
        while progressed:
            progressed = False
            for k in sched:
                i = cursors[k]
                li = bucket_lists[k]
                if i < len(li):
                    flat.append(li[i])
                    cursors[k] = i + 1
                    progressed = True

        # Rank sharding: balanced via striding (avoids empty shards)
        if self.num_replicas > 1:
            shard = flat[self.rank::self.num_replicas]
        else:
            shard = flat

        self._shard_indices = shard
