"""
+----------------------------------------------------------------------------------------------------------------------+
nntp_dm.py
Lightning data module for NeuroNTP sessions

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Loads the dataset view given with --ds_view, using the view class defined in the model configuration, and
    provides the train, validation and test data loaders. Before the first epoch, the clinical marker embeddings are
    initialised (if markers or notes are enabled), and priority sampling or random subsampling is applied to the view
    if requested. In continuous inference mode, the samples are streamed in temporal order with the continuous
    iterable class of the configuration.
+----------------------------------------------------------------------------------------------------------------------+
"""
import pytorch_lightning as pl
import logging
from torch.utils.data.distributed import DistributedSampler
from neuro_ntp.model.mmt.encoder.frz_sbert import FrozenSBERT
from neuro_ntp.model.ntp.nntp_misc import pretrain_collate_fn
from torch.utils.data import DataLoader
import neuro_ntp.misc.nntp_utils as nnutils
import torch

logger = logging.getLogger(__name__)

class PretrainingDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = args.config
        self.ds_view = None
        self.train_ds = self.val_ds = self.test_ds = None

    # ---------- one-time, CPU-only preprocessing (rank 0) ----------
    def prepare_data(self):
        # instantiate view class from the provided file using the configured view class
        self.ds_view = nnutils.instantiate(self.config['dataset_view_class'],
                                           dv_filepath=self.args.ds_view,
                                           use_cache=False)

        # initialize marker embeddings
        if self.config['add_markers'] or self.config['add_notes'] or self.config['decode_markers']:
            device = self.args.device
            if self.args.device != "cpu":
                device = "cuda:" + self.args.device.split(",")[0]
            logger.info(f"Initializing marker embeddings on device {device}...")
            sbert = FrozenSBERT(self.config['marker_encoder']).to(device)
            if self.args.reduced_vocab:
                self.ds_view.cfg['filter_keywords'] = self.config['filter_keywords']
                logger.info(f"REDUCED VOCABULARY, {self.ds_view.cfg['filter_keywords']}")
                self.ds_view.initialize_marker_embeddings(sbert, sbert.out_dim, batch_size=256, reduced_vocab=self.args.reduced_vocab)
            else:
                self.ds_view.initialize_marker_embeddings(sbert, sbert.out_dim, batch_size=256)
            if self.args.ext_markers:
                logger.info("Building extended marker embeddings...")
                self.ds_view.build_extended_marker_dict(batch_size=256)
            del sbert
            logger.info("Finished initializing marker embeddings...")

        if self.args.priority_sampling:
            self.ds_view.init_priority_sampling(
                event_frac=self.config['event_frac'],
                nonevent_frac=self.config['nonevent_frac']
            )
            if hasattr(self, "grouped_train_ds"):
                self.grouped_train_ds.rebuild()
            if hasattr(self, "grouped_val_ds"):
                self.grouped_val_ds.rebuild()
            if hasattr(self, "grouped_test_ds"):
                self.grouped_test_ds.rebuild()

        if self.args.random_subsampling:
            self.ds_view.init_random_subsampling(seed=self.args.random_sampling_seed, subset_size=self.args.random_subset_size,
                                                 balance_ratio=self.args.balance_ratio,
                                                 proximity_radius=self.args.proximity_radius)
            if self.ds_view is not None and hasattr(self.ds_view, "random_subsample"):
                self.ds_view.random_subsample()
            else:
                pass


    # ---------- per-rank setup ----------
    def setup(self, stage: str = None):
        if self.ds_view is None:
            # instantiate view class from the provided file using the configured view class
            self.ds_view = nnutils.instantiate(self.config['dataset_view_class'], dv_filepath=self.args.ds_view,
                                               use_cache=False)

        self.ds_view.update_channel_config(self.config)

        # re-apply runtime split config to loaded view.
        if 'split' in self.config:
            for key in ("split", "split_ratio", "split_def"):
                if key in self.config:
                    self.ds_view.cfg[key] = self.config[key]
            self.ds_view._split_samples()

        self.ds_view._init_split_views()

        # split datasets
        self.train_ds, self.val_ds, self.test_ds = self.ds_view.get_datasets()

        # optional logging
        if stage in (None, "fit", "test"):
            if self.train_ds and self.val_ds and self.test_ds:
                logger.info(
                    f"Dataset size (samples): {len(self.train_ds)} / {len(self.val_ds)} / {len(self.test_ds)}"
                )

    # ---------- dataloaders ----------
    def train_dataloader(self):
        logger.debug(f">> DM.train_dataloader called; batch_size={self.args.batch_size}")
        if getattr(self.args, "continuous_inference", False):
                logger.debug(">> DM.train_dataloader: continuous mode")
                assert self.args.batch_size == 1, "continuous_inference requires batch_size=1"
                assert self.args.num_workers == 0, "continuous_inference requires num_workers=0 (HDF caching)"

                ds = nnutils.instantiate(self.config['continuous_iterable_class'], ds_view=self.ds_view, split="train")

                return DataLoader(
                    ds,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    persistent_workers=False,
                    collate_fn=pretrain_collate_fn,
                )
        else:
            return self.init_dataloader(self.train_ds, type="train")

    def on_train_epoch_start(self):
        if hasattr(self, "grouped_train_ds"):
            self.grouped_train_ds.set_epoch(self.trainer.current_epoch)


    def val_dataloader(self):
        logger.debug(f">> DM.val_dataloader called; batch_size={self.args.batch_size}")
        return self.init_dataloader(self.val_ds, type="val")

    def test_dataloader(self):
        logger.debug(f">> DM.test_dataloader called; batch_size={self.args.batch_size}")

        # --- continuous mode (batch_size must be 1) ---
        if getattr(self.args, "continuous_inference", False):
            assert self.args.batch_size == 1, "continuous_inference requires batch_size=1"
            assert self.args.num_workers == 0, "continuous_inference requires num_workers=0 (HDF caching)"

            ds = nnutils.instantiate(self.config['continuous_iterable_class'], ds_view=self.ds_view, split="test")

            return DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                collate_fn=pretrain_collate_fn,
            )

        # --- standard Epi dataloader initialization ---
        return self.init_dataloader(self.test_ds, type="test")

    def init_dataloader(self, dataset, type):
        nnutils.set_seed(self.args.random_seed)
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 and type != "test" else None

        dl = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(sampler is None and type == "train"),
            num_workers=self.args.num_workers if type != "test" else 2,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=pretrain_collate_fn,
            drop_last=(type != "test" and sampler is not None),
            sampler=sampler
        )
        logger.info(f"<< {type.upper()} >> dataloader (batches): {len(dl)}")
        return dl


