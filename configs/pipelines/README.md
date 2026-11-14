# Pipeline configurations

A pipeline configuration describes one or more NeuroNTP sessions (`src/neuro_ntp/nntp.py`) that are run one after
another by `run_pipeline.py`. Each stage is translated into a plain `nntp.py` command line, which is printed before it
runs; a failing stage stops the pipeline.

```bash
python run_pipeline.py configs/pipelines/<pipeline>.yaml [--device N] [--set key=value ...] [--dry_run]
```

| Option | Effect |
|---|---|
| `--device N` | sets `CUDA_VISIBLE_DEVICES=N` for all stages (the stages then address it as device `0`) |
| `--set key=value` | overrides an entry of `vars` (repeatable) |
| `--dry_run` | prints the downloads and commands without running anything |

## Structure

```yaml
vars:                       # named values, referenced as ${name}
  ds_view: datasets/views/chbmit/chbmit_selfsup_10s_biot.h5
  save_path: models/finetune

downloads:                  # optional: local path -> URL, fetched before the first stage if missing
  models/pretrain/kuk/kuk_pretrained_mm.ckpt: https://...

common:                     # nntp.py arguments shared by all stages
  device: 0
  ds_view: ${ds_view}

stages:                     # run in order
  - name: stage1
    mode: finetune
    args:                   # nntp.py arguments of this stage, overriding common
      run_name: Example_${run_id}
      ckpt_path: models/pretrain/kuk/kuk_pretrained_mm.ckpt
      save_path: ${save_path}
      freeze_mmt: true
  - name: test
    mode: dws_test
    args:
      ckpt_path: ${stage1.best_ckpt}
```

- **`vars`** are resolved in order, so later entries can reference earlier ones. `${run_id}` (timestamp and random
  suffix) is always defined and keeps run names of repeated runs apart.
- **Arguments** map to the command line as `key: value` → `--key=value`, `key: true` → `--key`; `false` or `null`
  omits the argument, also one set in `common`. The available arguments are listed by
  `python src/neuro_ntp/nntp.py --help`.
- **Checkpoints of earlier stages:** after a stage in `pretrain`, `linprobe` or `finetune` mode, its checkpoint is
  available to later stages (and its directory as `${<stage>.ckpt_dir}`):
  - `linprobe`, `finetune`: the best checkpoint by the monitored metric, as `${<stage>.best_ckpt}`. It is looked up
    as `<save_path>/checkpoints/<run_name>*/best*.ckpt` (fine-tuning) or as
    `<save_path>/logs/<run_name>*/**/checkpoints/*.ckpt` (linear probing, `last*.ckpt` ignored).
  - `pretrain`: the checkpoint at the end of the training schedule, as `${<stage>.last_ckpt}`. It is looked up as
    `<save_path>/logs/<run_name>*/**/checkpoints/last*.ckpt` or, if not present, as the latest scheduled dump
    `<save_path>/checkpoints/<run_name>*/epoch_<N>.ckpt` (`dump_every`).

  A checkpoint is only defined if the match is unique. Including `${run_id}` in `run_name` keeps it unique.
- **Paths** are relative to the repository root, where all stages run.

## Provided pipelines

| Pipeline | Content |
|---|---|
| `tusz_pretrain.yaml` | pretraining from scratch on TUSZ |
| `chb_finetune_biot.yaml` | two-stage fine-tuning of the KUK-pretrained model on CHB-MIT (BIOT-compatible view) and downstream test |
| `chb_dws_test_biot.yaml` | downstream test of the released CHB-MIT fine-tuned model (BIOT-compatible view) |
| `siena_finetune.yaml` | two-stage fine-tuning of the KUK-pretrained model on Siena (balanced view) and downstream test (test view) |
| `siena_quickstart.yaml` | downstream test of the released Siena fine-tuned model (test view); see the quick start in the main README |

They can be copied and adapted to create further pipelines, e.g. for another dataset view or model configuration.