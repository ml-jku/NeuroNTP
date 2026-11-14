# NeuroNTP model configurations

A model configuration (.yaml) defines the architecture, the input modalities, the training objectives and the
dataset classes of a NeuroNTP session. It is passed to `nntp.py` with `--nntp_cfg` (in pipelines, as the `nntp_cfg`
argument of a stage), and its full content is written to the log at the start of each session.

The configurations are organised by next-token predictor backbone and regime. The provided configurations use the
xLSTM backbone:

```
configs/nntp/
└── xlstm/
    ├── pretrain/   pretraining on TUSZ (tusz_pretrain.yaml) and on the KUK dataset (kuk_pretrain.yaml, the
    │               configuration of the released KUK-pretrained model)
    └── finetune/   fine-tuning and downstream test on CHB-MIT and Siena
```

Each configuration also contains the settings of the other backbones (`transformer`, `lintrafo`, `mamba`); a
configuration for another backbone is obtained by changing `ntp_type`. The KUK dataset is not public, so
`kuk_pretrain.yaml` documents the released model and cannot be used for training.

Every parameter is commented in the files themselves. The groups below give an overview; for the meaning and the
admissible values of individual parameters, see the comments in the configuration files.

## Parameter groups

| Group | Parameters | Content |
|---|---|---|
| Architecture | `embedding_dim`, `max_seq_len`, `fusion_level`, `fuse_*` | token embedding size, sequence length, and whether tokens are formed per channel (`channel`) or fused per timestep (`timestep`, using the fusion transformer) |
| Next-token predictor | `ntp_type`, `trf_*`, `xlstm`, `mamba` | backbone selection (`xLSTM`, `transformer`, `lintrafo`, `mamba`) and its settings; only the section of the selected backbone is used |
| Modalities and tokens | `add_*`, `signal_channels` | which inputs are encoded as tokens (EEG, non-EEG signals, electrode locations, metadata, markers, end-of-segment token) and the signal channels to use |
| Signal encoder | `sigenc_*` | STFT front end, convolutional stem and channel-specific heads of the multimodal tokenizer |
| Metadata and text encoding | `tps_*`, `tts_*`, `seizure_len_cap`, `marker_*` | encoding of time past / time to seizure and of clinical markers |
| Multitask heads | `predict_*`, `decode_markers` | auxiliary prediction tasks trained alongside next-token prediction |
| Downstream tasks | `downstream_tasks`, `finetuning_coeff` | task-specific heads for fine-tuning: head class, where it is attached (`mmt` or `ntp`), and its arguments |
| Sampling and regularisation | `event_frac`, `nonevent_frac`, `signals_channel_drop_*`, `*_mask_prob`, `mask_sch_*`, dropout | priority sampling fractions, channel dropout, token masking and its schedule |
| Pretraining objectives | `pt_loss_*`, `pt_*_weights`, `pt_*quantiles`, ... | loss coefficients and loss settings of the primary and auxiliary objectives |
| Dataset classes | `core_dataset_class`, `dataset_view_class` | classes used to load the dataset view given with `--ds_view` |

Loss coefficients of the pretraining objectives:

| Parameter | Objective |
|---|---|
| `pt_loss_alfa` | next-token prediction (primary objective) |
| `pt_loss_beta` | token type |
| `pt_loss_gamma` | time past seizure |
| `pt_loss_delta` | time to seizure |
| `pt_loss_epsilon` | segment type |
| `pt_loss_zeta` | marker embedding reconstruction |

## Creating a configuration

Start from a provided configuration of the same regime. The dataset classes and `signal_channels` must
match the dataset view used in the session; the modality flags (`add_*`) should only enable inputs the dataset
provides. A fine-tuning configuration must be compatible with the architecture of the checkpoint it starts from.