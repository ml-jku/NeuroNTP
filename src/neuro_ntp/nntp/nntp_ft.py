"""
+----------------------------------------------------------------------------------------------------------------------+
nntp_ft.py
Fine-tuning and downstream testing of NeuroNTP

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Runs a fine-tuning session (--mode finetune) or a downstream test (--mode dws_test): builds NeuroNTP with the
    task-specific heads of the model configuration attached, restores the weights from --ckpt_path, and trains or
    tests it with PyTorch Lightning. During fine-tuning, the best model according to --model_select is saved to
    <save_path>/checkpoints/<run_name>/best-epoch=NN.ckpt, together with the last epoch (last.ckpt); training stops
    early on the validation loss unless --no_early_stop is given.
+----------------------------------------------------------------------------------------------------------------------+
"""
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from neuro_ntp.model.ntp.neuro_ntp import NeuroNTP
from neuro_ntp.nntp.nntp_dm import PretrainingDataModule
from neuro_ntp.nntp.nntp_pt import PretrainingLightningModel
import neuro_ntp.misc.nntp_utils as nnutils

import logging
logger = logging.getLogger(__name__)

def finetune(args, test_only=False):
    # console message logger to use
    logger = logging.getLogger()

    # create datamodule
    pretraining_datamodule = PretrainingDataModule(args)

    # instantiate the Neuro-NTP model
    args.config['downstream'] = True
    args.config['linprobe'] = False
    model = NeuroNTP(config=args.config)
    model.mm_tokenizer.token_masking = args.token_masking

    # lightning model
    lightning_model = PretrainingLightningModel(args, model)

    # logger and callbacks
    version = f"{args.run_name}-{args.dataset}-{args.lr}-{args.batch_size}"
    if args.wandb:
        tr_logger = WandbLogger(
            save_dir=f"{args.save_path}/logs/{args.run_name}_{args.run_uuid}",
            # version=version,
            name=version + f"_{np.random.randint(1000)}",
            settings=dict(_disable_stats=True),
        )
        tr_logger.log_hyperparams(vars(args))
    else:
        tr_logger = False

    modsel_mon, modsel_mode = args.model_select.split("|")
    logger.info(f"...model selection criterion: {modsel_mode}({modsel_mon})")
    checkpoint_cb = ModelCheckpoint(
        monitor=modsel_mon,
        mode=modsel_mode,
        save_top_k=1,
        save_last=True,
        #filename="nntp_ft-{epoch:02d}-{val_loss:.4f}",
        dirpath=f"{args.save_path}/checkpoints/{args.run_name}",
        filename="best-{epoch:02d}",
    ) if not test_only else None
    callbacks = [TQDMProgressBar(refresh_rate=10)]
    if not test_only:
        callbacks.append(checkpoint_cb)

        # early stopping with validation loss criterion
        if not args.no_early_stop:
            callbacks.append(EarlyStopping(monitor="val/loss", patience=args.patience, verbose=True, mode="min"))

    trainer = pl.Trainer(
        devices=[int(d) for d in args.device.split(",")] if args.device != "cpu" else 1,
        accelerator="gpu" if args.device != "cpu" else "cpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="bf16-mixed" if args.fp16 and torch.cuda.is_bf16_supported() else ("16-mixed" if args.fp16 else 32),
        benchmark=True,
        enable_checkpointing=not test_only,
        logger=tr_logger,
        log_every_n_steps=50,
        max_epochs=args.epochs,
        callbacks=callbacks,
        gradient_clip_val=0.75,
        sync_batchnorm=True,
        accumulate_grad_batches=2,
        reload_dataloaders_every_n_epochs=1,
    )

    # restoring a saved model
    if args.ckpt_path:
        nnutils.restore_from_ckpt(args.ckpt_path, pretraining_datamodule, lightning_model)

    # train the model
    if not test_only:
        logger.info("Starting finetuning on downstream tasks...")
        try:
            trainer.fit(lightning_model, datamodule=pretraining_datamodule)
            logger.info("Finished finetuning...")
        except KeyboardInterrupt:
            logger.info("Aborting finetuning - keyboard interrupt...")
    else:
        logger.info("Start evaluation on test set...")
        trainer.test(model=lightning_model, ckpt_path=None, datamodule=pretraining_datamodule)[0]
        logger.info("Finished evaluation on test set...")


