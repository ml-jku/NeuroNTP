from collections import defaultdict, deque
from typing import List, Hashable, Iterable, Optional
import random
import itertools
from torch.utils.data import IterableDataset, get_worker_info


class GroupedIterable(IterableDataset):
    """
    pre-batches dataset indices by a key (dataset.bucket_key(idx)).
    """
    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False,
        seed: int = 0,
        ddp_shard: bool = False,
    ):
        super().__init__()
        self.ds = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.ddp_shard = bool(ddp_shard)

        self._batches: List[List[int]] = []
        self.rebuild(epoch=0)

    # ---------- public helpers ----------
    def __len__(self) -> int:
        return len(self._batches)

    def rebuild(self, epoch: int = 0):
        """ recompute the pre-batched index lists """
        rng = random.Random(self.seed + int(epoch))

        # base index order (you can swap in your own sampler order if needed)
        indices: Iterable[int] = range(len(self.ds))

        # DDP rank/world sharding at build time
        if self.ddp_shard:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()
                    world = dist.get_world_size()
                    indices = itertools.islice(indices, rank, None, world)
            except Exception:
                pass

        # bucket indices by key
        buckets: dict[Hashable, deque[int]] = defaultdict(deque)
        for idx in indices:
            k = self.ds.bucket_key(idx)  # must be hashable and aligned to dataset indexing
            buckets[k].append(idx)

        # shuffle key order and within-bucket order
        keys = list(buckets.keys())
        if self.shuffle:
            rng.shuffle(keys)

        batches: List[List[int]] = []
        for k in keys:
            L = list(buckets[k])
            if self.shuffle:
                rng.shuffle(L)
            # slice into batches
            for i in range(0, len(L), self.batch_size):
                batch_idx = L[i:i + self.batch_size]
                if len(batch_idx) == self.batch_size or not self.drop_last:
                    batches.append(batch_idx)

        # shuffle batch order
        if self.shuffle:
            rng.shuffle(batches)

        self._batches = batches

    def set_epoch(self, epoch: int):
        """ compatibility helper; call this from your Lightning DataModule on epoch start."""
        self.rebuild(epoch=epoch)

    # ---------- iteration ----------
    def __iter__(self):
        # worker sharding happens at iteration time: each worker gets a disjoint slice of batches
        worker = get_worker_info()
        if worker is None or worker.num_workers <= 1:
            batch_iter = self._batches
        else:
            # round-robin split by worker id
            batch_iter = itertools.islice(self._batches, worker.id, None, worker.num_workers)

        for batch_idx in batch_iter:
            yield [self.ds[i] for i in batch_idx]
