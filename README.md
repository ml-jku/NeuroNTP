# NeuroNTP: A Generalizable Multimodal Foundation Model for Epilepsy

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-6f42c1)](https://neurips.cc/virtual/2026/poster/149129)
[![Paper](https://img.shields.io/badge/paper-OpenReview-8c1b13)](https://openreview.net/forum?id=seM8LkgKy6)
[![Weights](https://img.shields.io/badge/weights-download-2ea44f)](https://cloud.ml.jku.at/s/Fs3sz2R7oWs6b6C)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
<!-- ![Status](https://img.shields.io/badge/status-partial%20release-orange) -->

<!-- 
[![Quick start](https://img.shields.io/badge/quick%20start-evaluation-blue)](#quick-start-evaluating-a-fine-tuned-model-on-siena)
-->
> [!TIP]
> 🚀 To evaluate a released fine-tuned model on the Siena Scalp EEG Database with a few commands, see the [Quick start](#quick-start-evaluating-a-fine-tuned-model-on-siena).

---

NeuroNTP is a PyTorch framework for multimodal self-supervised learning on biosignals. It has two main components:

- **Multimodal tokenizer:** an extensible encoder that maps signals such as EEG, ECG and SpO2 into a shared embedding space.
- **Next-token predictor:** a causal sequence model, configurable as an xLSTM, Transformer, Linear Transformer or Mamba.

Pretraining is driven by next-token prediction, with auxiliary tasks in a multitask head acting as regularisers.

Within each timestep, the embeddings are arranged as a subsequence of channel-level tokens. This forms one continuous token stream, so the model learns to predict the next embedding from the history of channel-level tokens.

![NeuroNTP channel-level tokenization](./res/nntp_architecture_channel.jpg)

## Setting up the environment

### Installation

NeuroNTP is installed as a Python package (`neuro_ntp`) inside a dedicated conda environment. From the repository root, run:

```bash
conda env create -f environment.yml
conda activate neurontp
```

This creates the `neurontp` environment and installs the package in editable mode, together with all its dependencies. Activate the environment before running any of the preprocessing, training or evaluation scripts.

To reproduce the experimental environment exactly (Linux x86_64, CUDA 12.8), install the pinned versions of `requirements-lock.txt` instead:

```bash
conda create -n neurontp -c conda-forge -c nvidia python=3.11 "cuda-toolkit>=12.8,<12.9" ninja pip
conda activate neurontp
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

### CUDA build environment

Some CUDA kernels (e.g. the xLSTM sLSTM kernels) are compiled on first use. `nntp.env` points the compiler to the CUDA toolkit of the conda environment; adjust `TORCH_CUDA_ARCH_LIST` to your GPUs and source the file once per shell after activating the environment:

```bash
. ./nntp.env
```

The file also contains optional settings for clusters with a shared home directory and for compute nodes without internet access.

### Optional: Mamba backbone

The Mamba backbone requires `mamba-ssm`, which is compiled against the installed PyTorch and is therefore installed separately (the build can take a while):

```bash
conda activate neurontp
. ./nntp.env
pip install "mamba-ssm>=2.3" --no-build-isolation
```

## Preparing the datasets

NeuroNTP uses a self-supervised dataset implementation built on our `EpilepsiaCoreDataset` / `EpilepsiaDatasetView` classes, so the data must first be preprocessed into a compatible format.

The [data preparation guide](scripts/README.md) has preprocessing instructions for the TUH EEG Seizure Corpus, CHB-MIT and the Siena Scalp EEG Database. It also explains how to prepare your own dataset for use with NeuroNTP.

## Running NeuroNTP

### Folder structure

```
neuro-ntp/
├── run_pipeline.py      runner for pipeline configurations
├── nntp.env             CUDA build environment
├── configs/
│   ├── data/            custom core dataset and dataset view configurations (defaults are defined in the classes)
│   ├── nntp/            model configurations (xLSTM backbone)
│   └── pipelines/       pipeline configurations
├── datasets/            raw/, hdf5/, views/ (see the data preparation guide)
├── logs/
├── models/              checkpoints: pretrain/, finetune/; released weights in pretrain/kuk/
├── res/                 resources
├── scripts/             dataset preprocessing scripts
└── src/neuro_ntp/       package: data/, misc/, model/, nntp/ and the main entry point nntp.py
```

### Sessions: `nntp.py`

Each pretraining, fine-tuning or evaluation session is one run of `src/neuro_ntp/nntp.py`, configured by two main inputs: a model configuration (`--nntp_cfg`, see [configs/nntp](configs/nntp/README.md)) and a dataset view (`--ds_view`, created by the preprocessing scripts).

| Mode (`--mode`) | Purpose |
|---|---|
| `pretrain` | training from scratch, or continued pretraining from a checkpoint (`--ckpt_path`) |
| `linprobe` | linear probing via continued pretraining with a probing configuration |
| `evaluate` | evaluation on the test split with the default auxiliary heads (also zero-shot on another dataset) |
| `finetune` | fine-tuning with a task-specific head attached to the next-token predictor |
| `dws_test` | downstream test of a fine-tuned model |

Frequently used arguments:

| Argument | Description |
|---|---|
| `--nntp_cfg`, `--ds_view` | model configuration (.yaml) and dataset view (.h5) |
| `--ckpt_path` | checkpoint to start from (continued pretraining, fine-tuning, evaluation) |
| `--save_path`, `--run_name` | checkpoints are written to `<save_path>/checkpoints/<run_name>.../` |
| `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--lr_scheduler` | optimisation |
| `--freeze_mmt`, `--freeze_ntp`, `--freeze_all` | freeze the tokenizer, the next-token predictor or the whole model |
| `--priority_sampling`, `--random_subsampling` | sampling strategy of the training samples |
| `--wandb`, `--random_seed`, `--num_workers` | logging, reproducibility, data loading |

By default, performance is evaluated on the prediction at the last token of each sample. The full list of arguments is shown by `python src/neuro_ntp/nntp.py --help`.

At the end of an evaluation or downstream test, the logged metrics are listed with the prefix `test/`:

- `seg_type_*`: segment-type predictions of the auxiliary head (`evaluate` mode, linear probing)
- `<task>_*`: predictions of a downstream head, named after the task in the model configuration (e.g. `ntp_segtype_*` for the head attached to the next-token predictor)
- `*_ctx_*`: evaluated at the last token position, i.e. the reported performance (e.g. `test/ntp_segtype_ctx_auroc`); metrics without `_ctx_` are aggregated over all positions of the samples

### Pipelines: `run_pipeline.py`

Sessions are usually run through pipeline configurations, which collect the arguments of one or more consecutive sessions in a YAML file. The runner prints each `nntp.py` command before running it, passes the checkpoint of one stage on to the next, and downloads released weights referenced by a pipeline if they are not available locally. The configuration format and the provided pipelines (TUSZ pretraining, fine-tuning, evaluation) are described in [configs/pipelines](configs/pipelines/README.md).

Fine-tuning of the KUK-pretrained model on CHB-MIT, followed by the downstream test:

```bash
python run_pipeline.py configs/pipelines/chb_finetune_biot.yaml --device 0
```

Pretraining from scratch on TUSZ:

```bash
python run_pipeline.py configs/pipelines/tusz_pretrain.yaml --device 0
```

`--dry_run` prints the commands without running them. For long runs, start the pipeline in a `tmux` session, or detached with output to a log file:

```bash
nohup python -u run_pipeline.py configs/pipelines/tusz_pretrain.yaml --device 0 > logs/tusz_pretrain.log 2>&1 &
```

### Pretrained model weights

The weights of the model pretrained on the Kepler Universitätsklinikum (KUK) dataset are available [here](https://cloud.ml.jku.at/s/Fs3sz2R7oWs6b6C). Pipelines that reference them download them to `models/pretrain/kuk/` on first use.

### Multi-GPU training

Currently, the pretraining code is compatible with single GPU training only, multi-GPU is not supported.

---
## Quick start: evaluating a fine-tuned model on Siena
⏱️ **Estimated time:** ~45 min

This path runs the downstream test of a released NeuroNTP model, fine-tuned for seizure detection on the Siena Scalp EEG Database, on the held-out test patients of the reference protocol (PN16, PN17). It requires a CUDA GPU, and disk space for the raw data (~21 GB) and the HDF5 store (~13 GB, see the [data preparation guide](scripts/README.md)).

```bash
# 1. environment (see Setting up the environment)
conda env create -f environment.yml
conda activate neurontp
. ./nntp.env

# 2. download and verify Siena, convert it to HDF5 and create the dataset views
python scripts/download_data.py --datasets siena --out_root datasets/raw
python scripts/siena2hdf.py datasets/raw/siena/1.0.0 --out_hdf_root "$(pwd)/datasets/hdf5/siena" --n_jobs 8
python scripts/create_siena_ds.py --hdf_root datasets/hdf5/siena --out_dir datasets/views/siena --bal_template res/view_template/siena_selfsup_dv_bal_template.h5

# 3. downstream test; downloads the fine-tuned model to models/finetune/siena/ on first use
python run_pipeline.py configs/pipelines/siena_quickstart.yaml --device 0
```

- Downloading and converting the data takes the most time. If necessary, re-run `download_data.py` until it reports `OK` before converting (see the [data preparation guide](scripts/README.md)).
- If you already have a local copy of Siena v1.0.0, skip the download and pass its folder to `siena2hdf.py`; check it first with `download_data.py --verify_only` (see the data preparation guide).
- On the first run, the xLSTM CUDA kernels are compiled, which takes a few minutes.
- At the end of the run, the metrics are listed with the prefix `test/`. The reported performance is evaluated at the last token position (`*_ctx_*`), e.g. `test/ntp_segtype_ctx_auroc`.
- The quick start pipeline runs without Weights & Biases logging; enable it with `wandb: true` in the pipeline configuration.

To fine-tune the KUK-pretrained model on Siena yourself, run `configs/pipelines/siena_finetune.yaml`.

> [!TIP]
> 🚀 To do a similar downstream test on the CHB-MIT dataset, run `configs/pipelines/chb_dws_test_biot.yaml` (requires [preparing the CHB-MIT dataset](scripts/README.md#chb-mit-scalp-eeg-database)).

## Results on public benchmarks

Seizure detection on CHB-MIT and Siena Scalp EEG. NeuroNTP is pretrained multimodally on the KUK dataset, then fine-tuned and evaluated with biosignal-only input (no clinical markers, notes or TPS). Values are means in %; best per column in bold.

| Model | CHB-MIT BACC | CHB-MIT AUPRC | CHB-MIT AUROC | Siena BACC | Siena AUPRC | Siena AUROC |
|---|---:|---:|---:|---:|---:|---:|
| CNN-Transformer | 63.9 | 24.8 | 86.6 | 69.8 | 38.4 | 87.2 |
| BIOT | 70.7 | 32.8 | 87.6 | 73.5 | 38.1 | 90.3 |
| LaBraM-Base | 70.8 | 32.9 | 86.8 | 70.8 | 31.2 | 88.1 |
| CBraMod | 74.0 | 36.9 | 88.9 | 73.2 | 41.1 | 90.4 |
| EEGMamba | 74.5 | 38.9 | 89.4 | – | – | – |
| CSBrain | 72.6 | **51.6** | 89.2 | 76.6 | **48.7** | 90.8 |
| **NeuroNTP (ours)** | **84.7** | 45.4 | **93.8** | **87.5** | 46.6 | **93.6** |

Baseline results are taken from [CBraMod](https://arxiv.org/abs/2412.07236) (CHB-MIT) and [CSBrain](https://arxiv.org/abs/2506.23075) (Siena), and we follow their evaluation protocols. The paper has the full tables, including standard deviations and further baselines.

## Citation

If you found this work useful, please cite our paper:

```bibtex
@inproceedings{kovacs2026neurontp,
  title     = {{NeuroNTP} - A Generalizable Multimodal Foundation Model for Epilepsy},
  author    = {Kov{\'a}cs, J{\'o}zsef and Hauser, Amadeus and Gr{\"o}ppel, Gudrun and Narzt, Wolfgang and Seidl, Philipp},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=seM8LkgKy6}
}
```
