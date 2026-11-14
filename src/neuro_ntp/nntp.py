"""
+----------------------------------------------------------------------------------------------------------------------+
nntp.py
Entry point for NeuroNTP pretraining, fine-tuning and evaluation sessions

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Authors: Jozsef Kovacs (jozsef.kovacs@jku.at), Philipp Seidl (philipp.seidl@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Runs one session in the operating mode selected with --mode. Each session requires a NeuroNTP configuration file
    (--nntp_cfg) and a dataset view (--ds_view, created with the create_<dataset>_ds.py scripts). Modes starting from
    a trained model take the checkpoint via --ckpt_path.

    Modes:
        pretrain     training from scratch, or continued pretraining from a checkpoint
        linprobe     linear probing via continued pretraining, with a configuration file defining the probing setup
        evaluate     evaluation on the test split using the default auxiliary heads; also used for zero-shot
                     evaluation on another dataset
        finetune     fine-tuning with a task-specific head attached to the next-token predictor (NTP)
        dws_test     downstream test of a fine-tuned model

    By default, performance is evaluated on the prediction at the last token of each sample (see --eval_ctx_len,
    --eval_ts_last_k).

    The command line arguments are documented in the README; example pipelines for the different regimes are
    provided in scripts/.
+----------------------------------------------------------------------------------------------------------------------+
"""
import os
import logging
import argparse
import torch
import yaml
import uuid
from neuro_ntp.nntp.nntp_pt import pretrain
from neuro_ntp.nntp.nntp_ft import finetune
import neuro_ntp.misc.nntp_utils as nnutils
from importlib.metadata import version, PackageNotFoundError

try:
    NNTP_VERSION = version("neuro_ntp")
except PackageNotFoundError:   # running from source without installation
    NNTP_VERSION = "unknown"

torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()

    # -------------------------------------------------
    # mode of operation
    # -------------------------------------------------
    parser.add_argument("--mode", type=str, default="pretrain", help="operating modes: pretrain | linprobe | evaluate | finetune | dws_test")

    # -------------------------------------------------
    # runtime environment related arguments - general
    # -------------------------------------------------
    parser.add_argument("--device", type=str, default="0", help="devices to use, e.g. 0,1,2,3 or cpu default 0; currently supports single device")
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers")
    parser.add_argument("--wandb", action="store_true", help="use wandb logger")
    parser.add_argument("--run_name", type=str, default="unk", help="run name")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed to set for this run")
    parser.add_argument("--debug", action="store_true", help="sets logging level to debug")
    parser.add_argument("--matmul_precision", type=str, default="medium", help="precision: medium | highest")
    parser.add_argument("--fp16", action="store_true", help="use fp16 precision")

    # -------------------------------------------------
    # learning dynamics, loss-fn & scheduler
    # -------------------------------------------------
    parser.add_argument("--epochs", type=int, default=50, help="preset number of epochs to run")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (peak learning rate with cos_warmup)")
    parser.add_argument("--lr_warmup", type=float, default=0.15, help="fraction of training steps used for warmup (cos_warmup)")
    parser.add_argument("--lr_scheduler", type=str, default=None, help="learning rate scheduler: cos_warmup | None (constant)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--model_select", type=str, default="val/loss|min", help="monitored metric and mode for model selection")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience in finetuning mode")
    parser.add_argument("--no_early_stop", action="store_true", help="no early stopping in finetuning mode")

    # -------------------------------------------------
    # dataset related arguments
    # -------------------------------------------------
    parser.add_argument("--dataset", type=str, default="NeuroNTP", help="dataset")
    parser.add_argument("--priority_sampling", action="store_true", help="apply prioritized sampling")
    parser.add_argument("--random_subsampling", action="store_true", help="apply randomized subsampling")
    parser.add_argument("--random_sampling_seed", type=int, default=42, help="random seed for sampling")
    parser.add_argument("--random_subset_size", type=float, default=0.15, help="percentage of the entire set to randomly subsample")
    parser.add_argument("--balance_ratio", type=float, default=None, help="ictal samples ratio in balanced sampling mode (ds support required)")
    parser.add_argument("--proximity_radius", type=float, default=None, help="subsample within proximity radius from seizures (ds support required)")
    parser.add_argument("--continuous_inference", action="store_true", help="feeds continuous stream for inference (ds support required)")

    # -------------------------------------------------
    # NeuroNTP configuration and model weights saving
    # -------------------------------------------------
    parser.add_argument("--nntp_cfg", type=str, default=None, help="configuration file for NeuroNTP")
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="model checkpoint save path")
    parser.add_argument("--dump_every", type=int, default=0, help="dump model weights every n epochs")
    parser.add_argument("--ckpt_path", type=str, help="continue training from this checkpoint")

    # -------------------------------------------------
    # dataset view to use (preprocessed)
    # -------------------------------------------------
    parser.add_argument("--ds_view", type=str, help="dataset view (.h5) to use in the session")

    # -------------------------------------------------
    # finetuning related arguments
    # -------------------------------------------------
    parser.add_argument("--freeze_head", action="store_true", help="freeze the downstream heads")
    parser.add_argument("--freeze_mmt", action="store_true", help="freeze the multi-modal tokenizer")
    parser.add_argument("--freeze_ntp", action="store_true", help="freeze the next-token predictor")
    parser.add_argument("--freeze_all", action="store_true", help="freeze the entire model")
    parser.add_argument("--token_masking", action="store_true", help="apply token masking")

    # -------------------------------------------------
    # evaluation related arguments
    # -------------------------------------------------
    parser.add_argument("--only_last", action="store_true", help="in finetuning use only last subsequence token")
    parser.add_argument("--topk", type=int, default=None, help="top-k parameter for predicted marker retrieval")
    parser.add_argument("--ext_markers", action="store_true", help="use extended marker dictionary for reconstruction (KUK)")
    parser.add_argument("--reduced_vocab", action="store_true", help="use reduced vocabulary defined in dataset (KUK)")
    parser.add_argument("--dws_loss_scope", type=str, default="dense", help="loss scope in finetuning: dense | eos | last_eos")
    parser.add_argument("--threshold", type=float, default=0.5, help="value to use for threshold-dependent metrics")
    parser.add_argument("--eval_ctx_len", type=int, default=-1, help="the length of the context window for evaluation (default: full/last token)")
    parser.add_argument("--eval_ts_last_k", type=int, default=0, help="number of tokens for evaluation preceding EoS within window (default: 0 - only last token")
    parser.add_argument("--eval_ts_last_k_agg", type=str, default="mean", help="how to aggregate predicted probabilities within k-tokens: max | mean")

    # parse known and undeclared arguments
    args, unknown = parser.parse_known_args()

    # unique run identifier
    args.run_uuid = uuid.uuid4().hex[:8]

    # set seed for reproducibility
    nnutils.set_seed(args.random_seed)

    # logging configuration
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y.%m.%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    banner = f"NeuroNTP (next-token predictor) v{NNTP_VERSION} - JKU Institute for Machine Learning / LIT AI Lab"
    logger.info("+" + "-" * 98 + "+")
    logger.info("| " + banner.ljust(97) + "|")
    logger.info("+" + "-" * 98 + "+")

    # add unknown args to args
    for arg in unknown:
        setattr(args, arg.split("--")[-1].split("=")[0], arg.split("=")[-1])

    # report command line arguments in the log
    logger.info("command line arguments:")
    logger.info(args)

    # read the NeuroNTP configuration (main configuration for any operating mode)
    if args.nntp_cfg:
        with open(args.nntp_cfg, "r") as f:
            logger.info(f"NeuroNTP configuration file loaded: {args.nntp_cfg}")
            args.config = yaml.safe_load(f)
            args.config['cfg_modalities'] = nnutils.configured_modalities_as_list(args.config)
            logger.info("NeuroNTP configuration:\n%s", yaml.dump(args.config, sort_keys=False, default_flow_style=False))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision(args.matmul_precision)

    if args.mode == 'pretrain' or args.mode == 'linprobe':
        logger.info(f"...starting in {args.mode.upper()} mode...")
        pretrain(args)
    elif args.mode == 'evaluate':
        logger.info("...starting in EVALUATE mode (test only)...")
        pretrain(args, test_only=True)
    elif args.mode == 'finetune':
        logger.info("...starting in FINETUNING mode...")
        finetune(args)
    elif args.mode == 'dws_test':
        logger.info("...starting in DOWNSTREAM TEST mode...")
        args.mode = 'finetune'
        finetune(args, test_only=True)
